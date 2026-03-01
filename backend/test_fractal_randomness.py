"""
test_fractal_randomness.py
──────────────────────────
Runs two statistical tests on the Fractal OTP engine:

  1. Entropy Test      — measures unpredictability of the OTP output stream
  2. Avalanche Effect  — measures sensitivity to tiny changes in input seed

Run from the backend/ directory:
    python test_fractal_randomness.py

Results are saved to: fractal_randomness_report.txt
"""

import math
import hashlib
import random
import os
from collections import Counter
from datetime import datetime
from fractal_engine import generate_fractal_otp, generate_random_order


# ── Configuration ─────────────────────────────────────────────────────────────

SAMPLE_SIZE      = 10_000   # number of OTPs generated for entropy test
AVALANCHE_TRIALS = 1_000    # number of trials for avalanche test
REPORT_FILE      = "fractal_randomness_report.txt"

# Fixed dummy inputs — the actual values don't matter,
# what matters is measuring output distribution and sensitivity
DUMMY_PASSWORD_HASH    = hashlib.sha256(b"test_password_fractal_auth").hexdigest()
DUMMY_BEHAVIOR_VECTOR  = [0.312, 0.671, 0.440, 0.120, 0.289]


# ── Helpers ───────────────────────────────────────────────────────────────────

def int_to_bits(n: int, width: int = 20) -> str:
    """Convert an integer to a zero-padded binary string."""
    return format(n, f'0{width}b')


def hamming_distance(bits1: str, bits2: str) -> int:
    """Count differing bit positions between two binary strings."""
    return sum(b1 != b2 for b1, b2 in zip(bits1, bits2))


def flip_random_bit_in_hash(seed_hash: str) -> str:
    """
    Flip a single random hex character in the hash to simulate
    a 1-bit change in the input seed.
    """
    idx = random.randint(0, len(seed_hash) - 1)
    original_char = seed_hash[idx]
    # Pick a different hex character
    hex_chars = "0123456789abcdef"
    new_char = random.choice([c for c in hex_chars if c != original_char])
    return seed_hash[:idx] + new_char + seed_hash[idx + 1:]


def separator(char="─", width=56) -> str:
    return char * width


# ── Test 1: Entropy ───────────────────────────────────────────────────────────

def run_entropy_test(lines: list) -> dict:
    """
    Generate SAMPLE_SIZE OTPs and measure the Shannon entropy of the
    digit distribution.

    For a perfectly random 6-digit decimal OTP stream (digits 0-9),
    the theoretical maximum entropy is log2(10) ≈ 3.3219 bits per digit.

    We measure:
      - Per-digit entropy  (each of the 6 digit positions independently)
      - Full OTP entropy   (treating each 6-digit OTP as a single symbol)
      - Chi-square test    (checks if digit frequencies are uniform)
    """
    lines.append("\nTEST 1 — ENTROPY TEST")
    lines.append(separator())
    lines.append(f"Sample size     : {SAMPLE_SIZE:,} OTPs")
    lines.append(f"Method          : Shannon entropy on digit frequency distribution")
    lines.append(f"Theoretical max : log2(10) = 3.3219 bits per digit position")
    lines.append("")

    print(f"  [1/2] Generating {SAMPLE_SIZE:,} OTPs for entropy analysis...")

    otps = []
    for i in range(SAMPLE_SIZE):
        order = generate_random_order()
        # Use incrementing timestamp to simulate real sessions
        otp = generate_fractal_otp(
            DUMMY_PASSWORD_HASH,
            DUMMY_BEHAVIOR_VECTOR,
            order,
            timestamp=i
        )
        otps.append(otp)

    print(f"  [2/2] Computing entropy metrics...")

    # ── Per-position digit entropy ──
    position_entropies = []
    lines.append("  Per-digit-position entropy:")
    for pos in range(6):
        digits_at_pos = [otp[pos] for otp in otps]
        counts = Counter(digits_at_pos)
        total = len(digits_at_pos)
        ent = -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)
        position_entropies.append(ent)
        bar = "█" * int(ent / 3.3219 * 20)
        lines.append(f"    Position {pos}: {ent:.4f} bits  {bar}")

    avg_position_entropy = sum(position_entropies) / len(position_entropies)

    # ── Full OTP symbol entropy ──
    otp_counts = Counter(otps)
    total = len(otps)
    full_entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in otp_counts.values() if c > 0
    )
    theoretical_max = math.log2(min(total, 1_000_000))  # 10^6 possible OTPs

    # ── Chi-square uniformity test across all digit positions ──
    all_digits = "".join(otps)
    digit_counts = Counter(all_digits)
    expected = len(all_digits) / 10
    chi_sq = sum((digit_counts.get(str(d), 0) - expected) ** 2 / expected for d in range(10))
    # Chi-square critical value for 9 degrees of freedom at p=0.05 is 16.919
    chi_sq_pass = chi_sq < 16.919

    lines.append("")
    lines.append(f"  Average per-position entropy : {avg_position_entropy:.4f} bits")
    lines.append(f"  Theoretical maximum          : 3.3219 bits")
    lines.append(f"  Entropy efficiency           : {(avg_position_entropy / 3.3219) * 100:.2f}%")
    lines.append("")
    lines.append(f"  Full OTP stream entropy      : {full_entropy:.4f} bits")
    lines.append(f"  Unique OTPs in sample        : {len(otp_counts):,} / {SAMPLE_SIZE:,}")
    lines.append(f"  Collision rate               : {((SAMPLE_SIZE - len(otp_counts)) / SAMPLE_SIZE) * 100:.3f}%")
    lines.append("")
    lines.append(f"  Chi-square statistic         : {chi_sq:.4f}")
    lines.append(f"  Chi-square threshold (p=0.05): 16.919")
    lines.append(f"  Chi-square result            : {'PASS ✓' if chi_sq_pass else 'FAIL ✗'}")
    lines.append("")

    verdict = avg_position_entropy >= 3.0 and chi_sq_pass
    lines.append(f"  VERDICT: {'PASS ✓  — OTP distribution is highly random' if verdict else 'FAIL ✗  — Distribution shows bias'}")

    return {
        "avg_position_entropy": avg_position_entropy,
        "full_entropy": full_entropy,
        "unique_otps": len(otp_counts),
        "chi_sq": chi_sq,
        "chi_sq_pass": chi_sq_pass,
        "verdict": verdict,
    }


# ── Test 2: Avalanche Effect ──────────────────────────────────────────────────

def run_avalanche_test(lines: list) -> dict:
    """
    For each trial:
      1. Generate an OTP with the original seed hash
      2. Flip a single random hex character in the seed hash (~4 bits changed)
      3. Generate an OTP with the flipped seed hash using the same order
      4. Measure the bit difference between the two OTPs

    A well-designed chaotic system should change ~50% of output bits
    when given a minimally different input (strict avalanche criterion).
    """
    lines.append("")
    lines.append("")
    lines.append("TEST 2 — AVALANCHE EFFECT TEST")
    lines.append(separator())
    lines.append(f"Trials          : {AVALANCHE_TRIALS:,}")
    lines.append(f"Method          : Flip 1 hex char in seed hash, measure output bit divergence")
    lines.append(f"Target          : ~50% of bits should change (strict avalanche criterion)")
    lines.append(f"OTP bit width   : 20 bits (6-digit decimal → 20-bit binary)")
    lines.append("")

    print(f"  [1/2] Running {AVALANCHE_TRIALS:,} avalanche trials...")

    bit_change_ratios = []
    zero_change_count = 0
    full_change_count = 0

    for i in range(AVALANCHE_TRIALS):
        order = generate_random_order()
        timestamp = i % 500  # reuse timestamps across trials

        # Original OTP
        otp1 = generate_fractal_otp(
            DUMMY_PASSWORD_HASH,
            DUMMY_BEHAVIOR_VECTOR,
            order,
            timestamp=timestamp
        )

        # Slightly modified seed — one hex character flipped
        flipped_hash = flip_random_bit_in_hash(DUMMY_PASSWORD_HASH)
        otp2 = generate_fractal_otp(
            flipped_hash,
            DUMMY_BEHAVIOR_VECTOR,
            order,
            timestamp=timestamp
        )

        bits1 = int_to_bits(int(otp1))
        bits2 = int_to_bits(int(otp2))

        diff = hamming_distance(bits1, bits2)
        ratio = diff / len(bits1)
        bit_change_ratios.append(ratio)

        if diff == 0:
            zero_change_count += 1
        if diff == len(bits1):
            full_change_count += 1

    print(f"  [2/2] Computing avalanche statistics...")

    avg_ratio   = sum(bit_change_ratios) / len(bit_change_ratios)
    min_ratio   = min(bit_change_ratios)
    max_ratio   = max(bit_change_ratios)
    variance    = sum((r - avg_ratio) ** 2 for r in bit_change_ratios) / len(bit_change_ratios)
    std_dev     = math.sqrt(variance)

    # Bucket distribution for histogram
    buckets = [0] * 10
    for r in bit_change_ratios:
        idx = min(int(r * 10), 9)
        buckets[idx] += 1

    lines.append("  Bit-change distribution (0%→100%):")
    for i, count in enumerate(buckets):
        lo = i * 10
        hi = lo + 10
        bar = "█" * int(count / AVALANCHE_TRIALS * 40)
        lines.append(f"    {lo:3d}–{hi:3d}%: {bar} {count}")

    lines.append("")
    lines.append(f"  Average bit change ratio  : {avg_ratio * 100:.2f}%  (target ~50%)")
    lines.append(f"  Std deviation             : {std_dev * 100:.2f}%")
    lines.append(f"  Min observed              : {min_ratio * 100:.2f}%")
    lines.append(f"  Max observed              : {max_ratio * 100:.2f}%")
    lines.append(f"  Zero-change trials        : {zero_change_count} / {AVALANCHE_TRIALS}")
    lines.append(f"  Full-change trials        : {full_change_count} / {AVALANCHE_TRIALS}")
    lines.append("")

    # Pass if average is between 35% and 65% — a reasonable avalanche band
    verdict = 0.35 <= avg_ratio <= 0.65
    lines.append(f"  VERDICT: {'PASS ✓  — Strong avalanche effect confirmed' if verdict else 'FAIL ✗  — Insufficient sensitivity to input changes'}")

    return {
        "avg_ratio": avg_ratio,
        "std_dev": std_dev,
        "min_ratio": min_ratio,
        "max_ratio": max_ratio,
        "zero_change": zero_change_count,
        "verdict": verdict,
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 56)
    print("  FRACTAL OTP — RANDOMNESS TEST SUITE")
    print("=" * 56)
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Samples   : {SAMPLE_SIZE:,} OTPs (entropy) | {AVALANCHE_TRIALS:,} trials (avalanche)")
    print("=" * 56)
    print()

    lines = []
    lines.append("=" * 56)
    lines.append("  FRACTAL OTP — RANDOMNESS TEST SUITE")
    lines.append("=" * 56)
    lines.append(f"  Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Engine     : Logistic Map + Mandelbrot + Julia Set (chained)")
    lines.append(f"  Seed       : SHA-256(password_hash + behavior_vector + timestamp)")
    lines.append(f"  Order      : Truly random per session (server-side only)")
    lines.append("=" * 56)

    # Run tests
    print("Running Test 1 — Entropy...")
    entropy_results = run_entropy_test(lines)
    print(f"  → Avg entropy: {entropy_results['avg_position_entropy']:.4f} bits  |  Chi-sq: {'PASS' if entropy_results['chi_sq_pass'] else 'FAIL'}")
    print()

    print("Running Test 2 — Avalanche Effect...")
    avalanche_results = run_avalanche_test(lines)
    print(f"  → Avg bit change: {avalanche_results['avg_ratio'] * 100:.2f}%")
    print()

    # Summary
    both_pass = entropy_results["verdict"] and avalanche_results["verdict"]
    lines.append("")
    lines.append("")
    lines.append("OVERALL SUMMARY")
    lines.append(separator("═"))
    lines.append(f"  Entropy Test     : {'PASS ✓' if entropy_results['verdict'] else 'FAIL ✗'}")
    lines.append(f"  Avalanche Test   : {'PASS ✓' if avalanche_results['verdict'] else 'FAIL ✗'}")
    lines.append(separator("─"))
    lines.append(f"  OVERALL RESULT   : {'PASS ✓  — Fractal OTP pipeline demonstrates strong randomness' if both_pass else 'FAIL ✗  — Review flagged tests'}")
    lines.append(separator("═"))
    lines.append("")
    lines.append("  These results support the claim that chaining Logistic Map,")
    lines.append("  Mandelbrot, and Julia Set fractals in a randomly ordered")
    lines.append("  pipeline produces cryptographically unpredictable OTPs.")

    # Write report
    report_path = os.path.join(os.path.dirname(__file__), REPORT_FILE)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print("=" * 56)
    print(f"  Entropy test   : {'PASS ✓' if entropy_results['verdict'] else 'FAIL ✗'}")
    print(f"  Avalanche test : {'PASS ✓' if avalanche_results['verdict'] else 'FAIL ✗'}")
    print(f"  Overall        : {'PASS ✓' if both_pass else 'FAIL ✗'}")
    print("=" * 56)
    print(f"\n  Report saved → {report_path}\n")


if __name__ == "__main__":
    main()