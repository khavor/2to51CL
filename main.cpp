#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include <sndfile.h>
#include <fftw3.h>
#include <string>
#include <chrono>
#include <iomanip>
#include <windows.h>
#include <fstream>
#include <filesystem>

constexpr size_t FFT_SIZE = 4096;
constexpr size_t HOP_SIZE = FFT_SIZE / 2; // 50% overlap
constexpr double PI = 3.14159265358979323846;
const float LIMITER_THRESHOLD = 0.94f;
const float RECOVERY_COEFF = 0.995f;

struct SurroundParams {
    double center_width = 0.25;      // 0.0 = exact center, 1.0 = entire stereo image
    double center_slope = 2.0;      // Falloff slope: 1.0 = gentle, 2.0 = standard, >= 4.0 = steep
    double ambient_slope = 2.0;      // Falloff slope: 1.0 = gentle, 2.0 = standard, >= 4.0 = steep
    double ambient_width = 1.0;     // 0.0 = extreme panned sounds only, 1.0 = entire L/R channels
    double ambient_delay_ms = 80.0; // Delay in milliseconds for ambient channels (typical: 15-30ms)
    double lfe_cutoff_hz = 120.0;   // LFE low-pass cutoff in Hz

    // Individual gain trim parameters in Decibels (dB) for all 6 output channels
    double gain_L_db = 0.0;
    double gain_R_db = 0.0;
    double gain_C_db = 0.0;
    double gain_lfe_db = 0.0;
    double gain_Ls_db = 0.0;
    double gain_Rs_db = 0.0;
};

void ExtractResource(int resourceId, const wchar_t* filename) {
    HRSRC hResource = FindResourceW(NULL, MAKEINTRESOURCEW(resourceId), MAKEINTRESOURCEW(RT_RCDATA));
    if (!hResource) return;
    HGLOBAL hMemory = LoadResource(NULL, hResource);
    DWORD dwSize = SizeofResource(NULL, hResource);
    LPVOID lpAddress = LockResource(hMemory);

    std::ofstream file(filename, std::ios::out | std::ios::binary);
    file.write((char*)lpAddress, dwSize);
    file.close();
}

void printUsage() {
    std::cout << "Usage: 2to51CL --input|-i <input_path> --output|-o <output_path> [options]\n"
        << "Options:\n"
        << "  --cwidth|-cw       Center spectral width fraction (0.0 to 1.0) [Default: 0.25]\n"
        << "  --cslope|-cs       Center spectral slope fraction (0.5 to +4.0) [Default: 2.0]\n"
        << "  --lfeclp|-lp       LFE cutoff low-pass frequency in Hz [Default: 120.0]\n"
        << "  --awidth|-aw       Ambient width fraction (0.0 = extreme panned, 1.0 = entire L/R) [Default: 1.0]\n"
        << "  --aslope|-as       Ambient spectral slope fraction (0.5 to +4.0) [Default: 2.0]\n"
        << "  --adelay|-ad       Ambient delay buffer length in ms [Default: 80.0]\n"
        << "  --gainlr|-gf       Gain for left and right channels [Default: 0.0]\n"
        << "  --gainc|-gc        Gain for center channel [Default: 0.0]\n"
        << "  --gainlfe|-gl      Gain for LFE channel [Default: 0.0]\n"
        << "  --gainslsr|-gs     Gain for surround left and right channels [Default: 0.0]\n";
    std::cout << "Gains are dB (e.g., 0.0 = unity, -6dB = 0.5, -100.0 = muted)\n";
}

// ============================================================================
// Ring Buffer for Ambient Delay
// ============================================================================
class DelayLine {
private:
    std::vector<double> buf;
    size_t write_idx = 0;
    size_t delay_len = 0;

public:
    void init(size_t delay_samples) {
        delay_len = delay_samples;
        buf.assign(delay_len + HOP_SIZE, 0.0);
        write_idx = 0;
    }

    void process(const double* in, double* out, size_t n) {
        if (delay_len == 0) {
            std::copy(in, in + n, out);
            return;
        }
        size_t cap = buf.size();
        for (size_t i = 0; i < n; ++i) {
            size_t read_idx = (write_idx + cap - delay_len) % cap;
            out[i] = buf[read_idx];
            buf[write_idx] = in[i];
            write_idx = (write_idx + 1) % cap;
        }
    }
};

// ============================================================================
// AVX Intrinsic Helpers (__m256d)
// ============================================================================

inline void apply_window_avx(const double* in, const double* win, double* out, size_t n) {
    for (size_t i = 0; i < n; i += 4) {
        __m256d v_in = _mm256_loadu_pd(&in[i]);
        __m256d v_win = _mm256_loadu_pd(&win[i]);
        _mm256_storeu_pd(&out[i], _mm256_mul_pd(v_in, v_win));
    }
}

inline void synthesis_ola_avx(const double* ifft_out, const double* win, double* accum, double scale, size_t n) {
    __m256d v_scale = _mm256_set1_pd(scale);
    for (size_t i = 0; i < n; i += 4) {
        __m256d v_out = _mm256_loadu_pd(&ifft_out[i]);
        __m256d v_win = _mm256_loadu_pd(&win[i]);
        __m256d v_acc = _mm256_loadu_pd(&accum[i]);
        __m256d v_prod = _mm256_mul_pd(v_out, v_win);
        _mm256_storeu_pd(&accum[i], _mm256_fmadd_pd(v_prod, v_scale, v_acc));
    }
}

inline void sub_lfe_center_gain_avx(const double* orig, const double* lfe, const double* center, double* out, double gain, size_t n) {
    __m256d v_gain = _mm256_set1_pd(gain);
    for (size_t i = 0; i < n; i += 4) {
        __m256d v_orig = _mm256_loadu_pd(&orig[i]);
        __m256d v_lfe = _mm256_loadu_pd(&lfe[i]);
        __m256d v_cent = _mm256_loadu_pd(&center[i]);
        __m256d v_hp = _mm256_sub_pd(v_orig, v_lfe);
        __m256d v_res = _mm256_sub_pd(v_hp, v_cent);
        _mm256_storeu_pd(&out[i], _mm256_mul_pd(v_res, v_gain));
    }
}

inline void copy_with_gain_avx(const double* in, double* out, double gain, size_t n) {
    __m256d v_gain = _mm256_set1_pd(gain);
    for (size_t i = 0; i < n; i += 4) {
        __m256d v = _mm256_loadu_pd(&in[i]);
        _mm256_storeu_pd(&out[i], _mm256_mul_pd(v, v_gain));
    }
}

inline void subtract_avx(const double* ina, const double* inb, double* out, size_t n) {
    //__m256d v_gain = _mm256_set1_pd(gain);
    for (size_t i = 0; i < n; i += 4) {
        __m256d v1 = _mm256_loadu_pd(&ina[i]);
        __m256d v2 = _mm256_loadu_pd(&inb[i]);
        _mm256_storeu_pd(&out[i], _mm256_sub_pd(v1, v2));
    }
}

inline void load_deinterleave_4complex(const fftw_complex* ptr, __m256d& re, __m256d& im) {
    const double* p = reinterpret_cast<const double*>(ptr);
    __m256d v0 = _mm256_loadu_pd(p);
    __m256d v1 = _mm256_loadu_pd(p + 4);

    __m256d unpck_lo = _mm256_unpacklo_pd(v0, v1);
    __m256d unpck_hi = _mm256_unpackhi_pd(v0, v1);

    re = _mm256_permute4x64_pd(unpck_lo, _MM_SHUFFLE(3, 1, 2, 0));
    im = _mm256_permute4x64_pd(unpck_hi, _MM_SHUFFLE(3, 1, 2, 0));
}

inline void interleave_store_4complex(fftw_complex* ptr, __m256d re, __m256d im) {
    double* p = reinterpret_cast<double*>(ptr);

    __m256d unpck_lo = _mm256_unpacklo_pd(re, im);
    __m256d unpck_hi = _mm256_unpackhi_pd(re, im);

    __m256d v0 = _mm256_permute2f128_pd(unpck_lo, unpck_hi, 0x20);
    __m256d v1 = _mm256_permute2f128_pd(unpck_lo, unpck_hi, 0x31);

    _mm256_storeu_pd(p, v0);
    _mm256_storeu_pd(p + 4, v1);
}

// ============================================================================
// Core DSP Up-mixer
// ============================================================================

__declspec(noinline) bool process_audio(const char* in_file, const char* out_file, const SurroundParams& params) {
    SF_INFO in_info = { 0 };
    SNDFILE* snd_in = sf_open(in_file, SFM_READ, &in_info);
    if (!snd_in) {
        std::cerr << "Error opening input file: " << sf_strerror(nullptr) << std::endl;
        return false;
    }

    if (in_info.channels != 2) {
        std::cerr << "Error: Input file must be stereo (2 channels)." << std::endl;
        sf_close(snd_in);
        return false;
    }
    const sf_count_t totalFrames = in_info.frames;
    SF_INFO out_info = in_info;
    out_info.channels = 6;
    out_info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_24;

    SNDFILE* snd_out = sf_open(out_file, SFM_WRITE, &out_info);
    if (!snd_out) {
        std::cerr << "Error opening output file: " << sf_strerror(nullptr) << std::endl;
        sf_close(snd_in);
        return false;
    }

    const double g_lin_L = std::pow(10.0, params.gain_L_db / 20.0);
    const double g_lin_R = std::pow(10.0, params.gain_R_db / 20.0);
    const double g_lin_C = std::pow(10.0, params.gain_C_db / 20.0);
    const double g_lin_LFE = std::pow(10.0, params.gain_lfe_db / 20.0);
    const double g_lin_Ls = std::pow(10.0, params.gain_Ls_db / 20.0);
    const double g_lin_Rs = std::pow(10.0, params.gain_Rs_db / 20.0);

    const double sample_rate = static_cast<double>(in_info.samplerate);

    const size_t ambient_delay_samples = static_cast<size_t>(
        max(0.0, std::round(params.ambient_delay_ms * sample_rate / 1000.0))
        );
    DelayLine delay_line_Ls, delay_line_Rs;
    delay_line_Ls.init(ambient_delay_samples);
    delay_line_Rs.init(ambient_delay_samples);

    std::vector<double> window(FFT_SIZE);
    for (size_t i = 0; i < FFT_SIZE; ++i) {
        window[i] = std::sin((i + 0.5) * PI / FFT_SIZE);
    }

    const size_t spec_size = FFT_SIZE / 2 + 1;

    std::vector<double> lfe_weights(spec_size);
    for (size_t k = 0; k < spec_size; ++k) {
        double freq = static_cast<double>(k) * sample_rate / FFT_SIZE;
        double f_ratio = freq / max(10.0, params.lfe_cutoff_hz);
        lfe_weights[k] = 1.0 / std::sqrt(1.0 + std::pow(f_ratio, 8.0));
    }

    // FFTW Allocations
    double* in_L = fftw_alloc_real(FFT_SIZE);
    double* in_R = fftw_alloc_real(FFT_SIZE);
    fftw_complex* spec_L = fftw_alloc_complex(spec_size);
    fftw_complex* spec_R = fftw_alloc_complex(spec_size);

    fftw_complex* spec_LFE = fftw_alloc_complex(spec_size);
    fftw_complex* spec_C = fftw_alloc_complex(spec_size);
    fftw_complex* spec_AmbL = fftw_alloc_complex(spec_size);
    fftw_complex* spec_AmbR = fftw_alloc_complex(spec_size);

    double* out_LFE = fftw_alloc_real(FFT_SIZE);
    double* out_C = fftw_alloc_real(FFT_SIZE);
    double* out_AmbL = fftw_alloc_real(FFT_SIZE);
    double* out_AmbR = fftw_alloc_real(FFT_SIZE);

    fftw_plan p_fwd_L = fftw_plan_dft_r2c_1d((int)FFT_SIZE, in_L, spec_L, FFTW_ESTIMATE);
    fftw_plan p_fwd_R = fftw_plan_dft_r2c_1d((int)FFT_SIZE, in_R, spec_R, FFTW_ESTIMATE);

    fftw_plan p_inv_LFE = fftw_plan_dft_c2r_1d((int)FFT_SIZE, spec_LFE, out_LFE, FFTW_ESTIMATE);
    fftw_plan p_inv_C = fftw_plan_dft_c2r_1d((int)FFT_SIZE, spec_C, out_C, FFTW_ESTIMATE);
    fftw_plan p_inv_AmbL = fftw_plan_dft_c2r_1d((int)FFT_SIZE, spec_AmbL, out_AmbL, FFTW_ESTIMATE);
    fftw_plan p_inv_AmbR = fftw_plan_dft_c2r_1d((int)FFT_SIZE, spec_AmbR, out_AmbR, FFTW_ESTIMATE);

    std::vector<double> ola_LFE(FFT_SIZE, 0.0);
    std::vector<double> ola_C(FFT_SIZE, 0.0);
    std::vector<double> ola_AmbL(FFT_SIZE, 0.0);
    std::vector<double> ola_AmbR(FFT_SIZE, 0.0);
    //std::vector<double> ola_L_sub(FFT_SIZE, 0.0);
    //std::vector<double> ola_R_sub(FFT_SIZE, 0.0);

    std::vector<double> delay_L(FFT_SIZE, 0.0);
    std::vector<double> delay_R(FFT_SIZE, 0.0);

    std::vector<double> delayed_AmbL(HOP_SIZE);
    std::vector<double> delayed_AmbR(HOP_SIZE);
    std::vector<double> final_L(HOP_SIZE);
    std::vector<double> final_R(HOP_SIZE);
    std::vector<double> final_L_sub(HOP_SIZE);
    std::vector<double> final_R_sub(HOP_SIZE);
    std::vector<double> final_C(HOP_SIZE);
    std::vector<double> final_LFE(HOP_SIZE);
    std::vector<double> final_Ls(HOP_SIZE);
    std::vector<double> final_Rs(HOP_SIZE);

    std::vector<double> interleaved_in(HOP_SIZE * 2);
    std::vector<double> interleaved_out(HOP_SIZE * 6);

    // Buffers for artifact-suppression smoothing (Temporal & Spectral)
    std::vector<double> raw_amb_mask(spec_size, 0.0);
    std::vector<double> spec_smoothed_mask(spec_size, 0.0);
    std::vector<double> prev_amb_mask(spec_size, 0.0);

    const double ifft_scale = 1.0 / static_cast<double>(FFT_SIZE);
    const double eps = 1e-12;
    const double effective_width = max(0.005, params.center_width);
    const double slope_power = 2.0 * max(0.1, params.center_slope);
    const double aslope_power = 2.0 * max(0.1, params.ambient_slope);
    const double effective_amb_width = max(0.005, params.ambient_width);

    // Temporal smoothing coefficient: ~70ms time constant to eliminate ~22Hz frame flutter
    const double temporal_alpha = 0.70;

    const __m256d v_half = _mm256_set1_pd(0.5);
    const __m256d v_one = _mm256_set1_pd(1.0);
    const __m256d v_zero = _mm256_setzero_pd();
    const __m256d v_eps = _mm256_set1_pd(eps);
    const __m256d v_inv_width = _mm256_set1_pd(1.0 / effective_width);
    const __m256d v_abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));

    const size_t vec_bins = spec_size - (spec_size % 4);

    float currentLimiterGain = 1.0f;
    sf_count_t read_count = 0;
    sf_count_t framesProcessed = 0;
    bool is_first_block = true;
    auto last_update = std::chrono::steady_clock::now();
    float CValLast = 0.0;
    float LRValLast = 0.0;
    float LFEValLast = 0.0;
    float LRsValLast = 0.0;
    float MaxValLast = 0.0;

    while ((read_count = sf_readf_double(snd_in, interleaved_in.data(), HOP_SIZE)) > 0) {
        if (read_count < (sf_count_t)HOP_SIZE) {
            std::fill(interleaved_in.begin() + read_count * 2, interleaved_in.end(), 0.0);
        }

        std::copy(delay_L.begin() + HOP_SIZE, delay_L.end(), delay_L.begin());
        std::copy(delay_R.begin() + HOP_SIZE, delay_R.end(), delay_R.begin());

        for (size_t i = 0; i < HOP_SIZE; ++i) {
            delay_L[HOP_SIZE + i] = interleaved_in[i * 2 + 0];
            delay_R[HOP_SIZE + i] = interleaved_in[i * 2 + 1];
        }

        apply_window_avx(delay_L.data(), window.data(), in_L, FFT_SIZE);
        apply_window_avx(delay_R.data(), window.data(), in_R, FFT_SIZE);

        fftw_execute(p_fwd_L);
        fftw_execute(p_fwd_R);

        // ====================================================================
        // Pass 1: LFE, Center Extraction, and Raw Ambient Mask Calculation
        // ====================================================================
        for (size_t k = 0; k < vec_bins; k += 4) {
            __m256d v_L_re, v_L_im, v_R_re, v_R_im;
            load_deinterleave_4complex(&spec_L[k], v_L_re, v_L_im);
            load_deinterleave_4complex(&spec_R[k], v_R_re, v_R_im);

            // 1. LFE extraction
            __m256d v_lfe_w = _mm256_loadu_pd(&lfe_weights[k]);
            __m256d v_mono_re = _mm256_mul_pd(_mm256_add_pd(v_L_re, v_R_re), v_half);
            __m256d v_mono_im = _mm256_mul_pd(_mm256_add_pd(v_L_im, v_R_im), v_half);

            __m256d v_LFE_re = _mm256_mul_pd(v_mono_re, v_lfe_w);
            __m256d v_LFE_im = _mm256_mul_pd(v_mono_im, v_lfe_w);

            // 2. High-passed stereo
            __m256d v_L_hp_re = _mm256_sub_pd(v_L_re, v_LFE_re);
            __m256d v_L_hp_im = _mm256_sub_pd(v_L_im, v_LFE_im);
            __m256d v_R_hp_re = _mm256_sub_pd(v_R_re, v_LFE_re);
            __m256d v_R_hp_im = _mm256_sub_pd(v_R_im, v_LFE_im);

            // Store LFE & HP signals
            interleave_store_4complex(&spec_LFE[k], v_LFE_re, v_LFE_im);
            interleave_store_4complex(&spec_AmbL[k], v_L_hp_re, v_L_hp_im); // temporarily hold L_hp
            interleave_store_4complex(&spec_AmbR[k], v_R_hp_re, v_R_hp_im); // temporarily hold R_hp

            // 3. Center extraction
            __m256d v_P_L = _mm256_fmadd_pd(v_L_hp_re, v_L_hp_re, _mm256_mul_pd(v_L_hp_im, v_L_hp_im));
            __m256d v_P_R = _mm256_fmadd_pd(v_R_hp_re, v_R_hp_re, _mm256_mul_pd(v_R_hp_im, v_R_hp_im));
            __m256d v_mag_L = _mm256_sqrt_pd(v_P_L);
            __m256d v_mag_R = _mm256_sqrt_pd(v_P_R);

            __m256d v_diff = _mm256_and_pd(_mm256_sub_pd(v_mag_L, v_mag_R), v_abs_mask);
            __m256d v_sum = _mm256_add_pd(_mm256_add_pd(v_mag_L, v_mag_R), v_eps);
            __m256d v_pan = _mm256_div_pd(v_diff, v_sum);

            __m256d v_center_mask;
            if (params.center_width >= 0.999) {
                v_center_mask = v_one;
            }
            else {
                __m256d v_norm_pan = _mm256_mul_pd(v_pan, v_inv_width);
                alignas(32) double npan[4];
                alignas(32) double cmask[4];
                _mm256_storeu_pd(npan, v_norm_pan);

                cmask[0] = 1.0 / (1.0 + std::pow(npan[0], slope_power));
                cmask[1] = 1.0 / (1.0 + std::pow(npan[1], slope_power));
                cmask[2] = 1.0 / (1.0 + std::pow(npan[2], slope_power));
                cmask[3] = 1.0 / (1.0 + std::pow(npan[3], slope_power));

                v_center_mask = _mm256_loadu_pd(cmask);
            }

            __m256d v_sum_hp_re = _mm256_mul_pd(_mm256_add_pd(v_L_hp_re, v_R_hp_re), v_half);
            __m256d v_sum_hp_im = _mm256_mul_pd(_mm256_add_pd(v_L_hp_im, v_R_hp_im), v_half);
            __m256d v_C_re = _mm256_mul_pd(v_sum_hp_re, v_center_mask);
            __m256d v_C_im = _mm256_mul_pd(v_sum_hp_im, v_center_mask);
            interleave_store_4complex(&spec_C[k], v_C_re, v_C_im);

            // 4. Raw Ambient Mask
            __m256d v_dot = _mm256_fmadd_pd(v_L_hp_re, v_R_hp_re, _mm256_mul_pd(v_L_hp_im, v_R_hp_im));
            __m256d v_tot_pwr = _mm256_add_pd(_mm256_add_pd(v_P_L, v_P_R), v_eps);
            __m256d v_icc = _mm256_div_pd(_mm256_add_pd(v_dot, v_dot), v_tot_pwr);

            __m256d v_pos_icc = _mm256_max_pd(v_zero, v_icc);
            // Linear transition (1.0 - pos_icc) avoids the sqrt(x) infinite derivative artifact
            __m256d v_diff_icc = _mm256_max_pd(v_zero, _mm256_sub_pd(v_one, v_pos_icc));

            alignas(32) double r_pan[4];
            alignas(32) double r_diff[4];
            _mm256_storeu_pd(r_pan, v_pan);
            _mm256_storeu_pd(r_diff, v_diff_icc);

            for (int b = 0; b < 4; ++b) {
                double pmask = 1.0;
                if (params.ambient_width < 0.999) {
                    double dist = (1.0 - r_pan[b]) / effective_amb_width;
                    pmask = (dist > 10.0) ? 0.0 : 1.0 / (1.0 + std::pow(dist, aslope_power));
                }
                raw_amb_mask[k + b] = r_diff[b] * pmask;
            }
        }

        // Scalar Tail for Pass 1
        for (size_t k = vec_bins; k < spec_size; ++k) {
            double L_re = spec_L[k][0], L_im = spec_L[k][1];
            double R_re = spec_R[k][0], R_im = spec_R[k][1];

            double mono_re = 0.5 * (L_re + R_re);
            double mono_im = 0.5 * (L_im + R_im);
            spec_LFE[k][0] = mono_re * lfe_weights[k];
            spec_LFE[k][1] = mono_im * lfe_weights[k];

            double L_hp_re = L_re - spec_LFE[k][0];
            double L_hp_im = L_im - spec_LFE[k][1];
            double R_hp_re = R_re - spec_LFE[k][0];
            double R_hp_im = R_im - spec_LFE[k][1];

            spec_AmbL[k][0] = L_hp_re;
            spec_AmbL[k][1] = L_hp_im;
            spec_AmbR[k][0] = R_hp_re;
            spec_AmbR[k][1] = R_hp_im;

            double P_L = L_hp_re * L_hp_re + L_hp_im * L_hp_im;
            double P_R = R_hp_re * R_hp_re + R_hp_im * R_hp_im;
            double mag_L = std::sqrt(P_L);
            double mag_R = std::sqrt(P_R);

            double pan = std::abs(mag_L - mag_R) / (mag_L + mag_R + eps);
            double cmask = (params.center_width >= 0.999) ? 1.0 : 1.0 / (1.0 + std::pow(pan / effective_width, slope_power));

            spec_C[k][0] = 0.5 * (L_hp_re + R_hp_re) * cmask;
            spec_C[k][1] = 0.5 * (L_hp_im + R_hp_im) * cmask;

            double dot = L_hp_re * R_hp_re + L_hp_im * R_hp_im;
            double icc = (2.0 * dot) / (P_L + P_R + eps);
            double diff_icc = max(0.0, 1.0 - max(0.0, icc));

            double pmask = 1.0;
            if (params.ambient_width < 0.999) {
                double dist = (1.0 - pan) / effective_amb_width;
                pmask = (dist > 10.0) ? 0.0 : 1.0 / (1.0 + std::pow(dist, aslope_power));
            }
            raw_amb_mask[k] = diff_icc * pmask;
        }

        // ====================================================================
        // Pass 2: Spectral Smoothing (removes musical noise / comb filtering)
        // ====================================================================
        spec_smoothed_mask[0] = 0.75 * raw_amb_mask[0] + 0.25 * raw_amb_mask[1];
        for (size_t k = 1; k < spec_size - 1; ++k) {
            spec_smoothed_mask[k] = 0.25 * raw_amb_mask[k - 1] + 0.50 * raw_amb_mask[k] + 0.25 * raw_amb_mask[k + 1];
        }
        spec_smoothed_mask[spec_size - 1] = 0.75 * raw_amb_mask[spec_size - 1] + 0.25 * raw_amb_mask[spec_size - 2];

        // ====================================================================
        // Pass 3: Temporal Smoothing (removes frame-to-frame warble & flutter)
        // ====================================================================
        for (size_t k = 0; k < spec_size; ++k) {
            double smoothed = temporal_alpha * prev_amb_mask[k] + (1.0 - temporal_alpha) * spec_smoothed_mask[k];
            prev_amb_mask[k] = smoothed;

            // Apply final smoothed mask to ambient frequency components
            spec_AmbL[k][0] *= smoothed;
            spec_AmbL[k][1] *= smoothed;
            spec_AmbR[k][0] *= smoothed;
            spec_AmbR[k][1] *= smoothed;
        }

        // Inverse FFTs
        fftw_execute(p_inv_LFE);
        fftw_execute(p_inv_C);
        fftw_execute(p_inv_AmbL);
        fftw_execute(p_inv_AmbR);

        // Synthesis windowing & Overlap-Add (AVX)
        synthesis_ola_avx(out_LFE, window.data(), ola_LFE.data(), ifft_scale, FFT_SIZE);
        synthesis_ola_avx(out_C, window.data(), ola_C.data(), ifft_scale, FFT_SIZE);
        synthesis_ola_avx(out_AmbL, window.data(), ola_AmbL.data(), ifft_scale, FFT_SIZE);
        synthesis_ola_avx(out_AmbR, window.data(), ola_AmbR.data(), ifft_scale, FFT_SIZE);

        float CVal;
        float LVal;
        float RVal;
        float LFEVal;
        float LsVal;
        float RsVal;
        float LRVal;
        float LRsVal;
        float OutMax;


        if (is_first_block) {
            is_first_block = false;
        }
        else {

            delay_line_Ls.process(ola_AmbL.data(), delayed_AmbL.data(), HOP_SIZE); //delay AmbL
            delay_line_Rs.process(ola_AmbR.data(), delayed_AmbR.data(), HOP_SIZE); //delay AmbR

            sub_lfe_center_gain_avx(delay_L.data(), ola_LFE.data(), ola_C.data(), final_L.data(), 1.0, HOP_SIZE);
            sub_lfe_center_gain_avx(delay_R.data(), ola_LFE.data(), ola_C.data(), final_R.data(), 1.0, HOP_SIZE);

            subtract_avx(final_L.data(), ola_AmbL.data(), final_L_sub.data(), HOP_SIZE); //from final_L subtract AmbL >> final_L_sub
            subtract_avx(final_R.data(), ola_AmbR.data(), final_R_sub.data(), HOP_SIZE); //from final_R subtract AmbR >> final_R_sub


            copy_with_gain_avx(ola_C.data(), final_C.data(), g_lin_C, HOP_SIZE);
            copy_with_gain_avx(ola_LFE.data(), final_LFE.data(), g_lin_LFE, HOP_SIZE);
            copy_with_gain_avx(delayed_AmbL.data(), final_Ls.data(), g_lin_Ls, HOP_SIZE);
            copy_with_gain_avx(delayed_AmbR.data(), final_Rs.data(), g_lin_Rs, HOP_SIZE);

            for (size_t i = 0; i < HOP_SIZE; ++i) {
                CVal = std::max<float>(std::abs(ola_C[i]), CValLast);
                LVal = std::abs(final_L_sub[i]);
                 RVal = std::abs(final_R_sub[i]);
                 LFEVal = std::max<float>(std::abs(ola_LFE[i]), LFEValLast);
                 LsVal = std::abs(ola_AmbL[i]);
                 RsVal = std::abs(ola_AmbR[i]);
                 LRVal = std::max<float>(std::max<float>(LVal, RVal), LRValLast);
                 LRsVal = std::max<float>(std::max<float>(LsVal, RsVal), LRsValLast);
                 CValLast = CVal;
                 LFEValLast = LFEVal;
                 LRValLast = LRVal;
                 LRsValLast = LRsVal;

                float maxVal = std::abs(final_C[i]);
                //maxVal = std::max(maxVal, std::abs(final_Ls[i]));
                // Replace this line:
                // maxVal = std::max(maxVal, std::abs(final_Ls[i]));
                // With this:
                maxVal = std::max<float>(maxVal, std::abs(final_L_sub[i]) * g_lin_L);
                maxVal = std::max<float>(maxVal, std::abs(final_R_sub[i]) * g_lin_R);
                maxVal = std::max<float>(maxVal, std::abs(final_Ls[i]));
                maxVal = std::max<float>(maxVal, std::abs(final_Rs[i]));
                maxVal = std::max<float>(maxVal, std::abs(final_LFE[i]));

                if (maxVal * currentLimiterGain > LIMITER_THRESHOLD) {
                    currentLimiterGain = LIMITER_THRESHOLD / (maxVal + 1e-6f);
                }
                maxVal *= currentLimiterGain;
                maxVal = std::max<float>(maxVal, MaxValLast);
                MaxValLast = maxVal;
                OutMax = maxVal;
                interleaved_out[i * 6 + 0] = final_L_sub[i] * currentLimiterGain * g_lin_L;
                interleaved_out[i * 6 + 1] = final_R_sub[i] * currentLimiterGain * g_lin_R;
                interleaved_out[i * 6 + 2] = final_C[i] * currentLimiterGain;
                interleaved_out[i * 6 + 3] = final_LFE[i] * currentLimiterGain;
                interleaved_out[i * 6 + 4] = final_Ls[i] * currentLimiterGain;
                interleaved_out[i * 6 + 5] = final_Rs[i] * currentLimiterGain;
                if (currentLimiterGain < 1.0f) {
                    currentLimiterGain += (1.0f - currentLimiterGain) * (1.0f - RECOVERY_COEFF);
                }
            }

            sf_writef_double(snd_out, interleaved_out.data(), HOP_SIZE);
        }

        auto shift_ola = [](std::vector<double>& buf) {
            std::copy(buf.begin() + HOP_SIZE, buf.end(), buf.begin());
            std::fill(buf.begin() + HOP_SIZE, buf.end(), 0.0);
        };
        shift_ola(ola_LFE);
        shift_ola(ola_C);
        shift_ola(ola_AmbL);
        shift_ola(ola_AmbR);

        framesProcessed += read_count;
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count() > 250 || framesProcessed == totalFrames) {
            double percent = (totalFrames > 0) ? (100.0 * framesProcessed / totalFrames) : 0.0;
            std::cout << "\rProcessing: "  << "(" << static_cast<int>(percent) << "%) completed - Output Max (Max) Pre-Gain Max (L/R) (C) (LFE) (Ls/Rs): " << std::fixed << std::setprecision(2) << OutMax << " " << LRVal << " " << CVal << " " << LFEVal << " " << LRsVal << std::flush;
            last_update = now;
        }
    }

    fftw_destroy_plan(p_fwd_L);
    fftw_destroy_plan(p_fwd_R);
    fftw_destroy_plan(p_inv_LFE);
    fftw_destroy_plan(p_inv_C);
    fftw_destroy_plan(p_inv_AmbL);
    fftw_destroy_plan(p_inv_AmbR);

    fftw_free(in_L);      fftw_free(in_R);
    fftw_free(spec_L);    fftw_free(spec_R);
    fftw_free(spec_LFE);  fftw_free(spec_C);
    fftw_free(spec_AmbL); fftw_free(spec_AmbR);
    fftw_free(out_LFE);   fftw_free(out_C);
    fftw_free(out_AmbL);  fftw_free(out_AmbR);

    sf_close(snd_in);
    sf_close(snd_out);

    return true;
}

int main(int argc, char* argv[]) {
    if (!std::filesystem::exists("sndfile.dll")) {
        ExtractResource(201, L"sndfile.dll");
    }
    if (!std::filesystem::exists(L"libfftw3-3.dll")) {
        ExtractResource(202, L"libfftw3-3.dll");
    }
    if (argc < 3) {
        printUsage();
        return 1;
    }
    const char* in_path = nullptr;
    const char* out_surround_path = nullptr;

    SurroundParams params;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--input" || arg == "-i") && i + 1 < argc) in_path = argv[++i];
        else if ((arg == "--output" || arg == "-o") && i + 1 < argc) out_surround_path = argv[++i];
        else if ((arg == "--cwidth" || arg == "-cw") && i + 1 < argc) params.center_width = std::stod(argv[++i]);
        else if ((arg == "--cslope" || arg == "-cs") && i + 1 < argc) params.center_slope = std::stod(argv[++i]);
        else if ((arg == "--lfecutoff" || arg == "-lc") && i + 1 < argc) params.lfe_cutoff_hz = std::stod(argv[++i]);
        else if ((arg == "--awidth" || arg == "-aw") && i + 1 < argc) params.ambient_width = std::stod(argv[++i]);
        else if ((arg == "--aslope" || arg == "-as") && i + 1 < argc) params.ambient_slope = std::stod(argv[++i]);
        else if ((arg == "--adelay" || arg == "-ad") && i + 1 < argc) params.ambient_delay_ms = std::stod(argv[++i]);
        else if ((arg == "--gainlr" || arg == "-gf") && i + 1 < argc) params.gain_L_db = std::stod(argv[++i]);
        else if ((arg == "--gainc" || arg == "-gc") && i + 1 < argc) params.gain_C_db = std::stod(argv[++i]);
        else if ((arg == "--gainlfe" || arg == "-gl") && i + 1 < argc) params.gain_lfe_db = std::stod(argv[++i]);
        else if ((arg == "--gainslsr" || arg == "-gs") && i + 1 < argc) params.gain_Ls_db = std::stod(argv[++i]);
        else if (arg == "--help" || arg == "-h") { printUsage(); return 0; }
    }
    params.gain_R_db = params.gain_L_db;
    params.gain_Rs_db = params.gain_Ls_db;

    if (!in_path || !out_surround_path) {
        std::cerr << "Error: Input and output paths are required." << std::endl;
        printUsage();
        return 1;
    }

    std::cout << "Processing Audio with 4-Bin AVX SIMD Acceleration..." << std::endl;
    std::cout << " - Center Width       : " << params.center_width << std::endl;
    std::cout << " - Center Slope       : " << params.center_slope << std::endl;
    std::cout << " - Ambient Width      : " << params.ambient_width
        << (params.ambient_width >= 0.999 ? " (entire L/R channels)" : " (restricted to extreme pans)") << std::endl;
    std::cout << " - Ambient Slope      : " << params.center_slope << std::endl;
    std::cout << " - Ambient Delay      : " << params.ambient_delay_ms << " ms" << std::endl;
    std::cout << " - LFE Cutoff         : " << params.lfe_cutoff_hz << " Hz" << std::endl;
    std::cout << " - Gains (dB) [LR/C/LFE/LsRs]: "
        << params.gain_L_db << " / "
        << params.gain_C_db << " / "
        << params.gain_lfe_db << " / "
        << params.gain_Rs_db << " dB" << std::endl;
    std::cout << "input: " << in_path << " output: " << out_surround_path << std::endl;

    if (process_audio(in_path, out_surround_path, params)) {
        std::cout << "\nProcessing completed successfully." << std::endl;
    }
    else {
        std::cerr << "\nProcessing failed." << std::endl;
        return 1;
    }

    return 0;
}
//Compile with: cl /EHsc /O2 /arch:AVX2 /I"include" main.cpp resources.res /link /LIBPATH:"lib" delayimp.lib sndfile.lib libfftw3-3.lib /DELAYLOAD:dll\sndfile.dll /DELAYLOAD:dll\libfftw3-3.dll /OUT:2to51CL.exe
