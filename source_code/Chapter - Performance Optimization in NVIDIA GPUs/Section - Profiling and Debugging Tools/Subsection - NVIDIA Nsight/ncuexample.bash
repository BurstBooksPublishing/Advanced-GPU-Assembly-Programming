ncu --target-processes all \
    --metrics sm_efficiency,achieved_occupancy,ipc,gld_throughput,gst_throughput \
    --launch-skip 1 --launch-count 5 \
    --kernel-name-base "myKernel" \
    ./my_app        # run app and collect kernel SASS metrics
# comments: --metrics picks hardware counters; --launch-skip/count focuses profiling scope.