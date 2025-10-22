import math

# expected_event_rate: events per second at hotspot (estimate)
# target_samples: desired number of samples for that hotspot
# run_time: total kernel runtime in seconds
def recommend_interval(expected_event_rate, target_samples, run_time):
    # required probability per sample to get target_samples on average
    p = target_samples / run_time  # events per second -> samples per second target
    # using p ~ f * dt -> dt ~ p / f
    dt = p / expected_event_rate
    # convert to microseconds for user convenience
    return max(dt, 1e-6) * 1e6  # microseconds

def sample_error(observed_k, n):
    p_hat = observed_k / n
    se = math.sqrt(p_hat * (1 - p_hat) / n)
    return p_hat, se

# example usage
if __name__ == "__main__":
    f = 1e6        # 1e6 events/sec at hotspot (estimate)
    target_n = 1000
    runtime = 10.0 # seconds
    dt_us = recommend_interval(f, target_n, runtime)
    print("# sample every {:.1f} us".format(dt_us))   # user sets timer accordingly
    # after run: compute error
    p_hat, se = sample_error(200, 20000)  # example: 200 hits out of 20k samples
    print("p_hat {:.4e}, SE {:.4e}".format(p_hat, se))