#include <stdio.h>
#include <stdlib.h>

/* Compute peak and sustained bandwidth for HBM-like stacks.
   All rates expressed in Gbits/s (Gbps) for r. */
int main(int argc, char **argv) {
    int S = 4;                 // number of HBM stacks (example)
    int C = 8;                 // channels per stack (HBM2 typical)
    int B_channel = 128;       // bits per channel
    double r = 1.6;            // per-pin data rate in Gbits/s
    double eta = 0.75;         // sustained efficiency factor (0..1)

    // peak per-stack (GB/s)
    double peak_stack = (C * B_channel * r) / 8.0 / 1000.0; // GB/s
    double peak_total = S * peak_stack;

    double sust_total = eta * peak_total;

    printf("Per-stack peak: %.3f GB/s\n", peak_stack);
    printf("Total peak    : %.3f GB/s\n", peak_total);
    printf("Sustained (eta=%.2f): %.3f GB/s\n", eta, sust_total);

    // show how many concurrent 32-bit loads per second at sustained rate
    double bytes_per_load = 4.0;
    double loads_per_sec = (sust_total * 1e9) / bytes_per_load;
    printf("Equivalent 32-bit loads/sec: %.3e\n", loads_per_sec);
    return 0;
}