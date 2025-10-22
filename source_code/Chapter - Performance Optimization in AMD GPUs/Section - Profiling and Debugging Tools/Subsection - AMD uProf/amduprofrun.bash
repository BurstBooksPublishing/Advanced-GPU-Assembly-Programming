# start uProf collection into a session directory (CLI name may be amduprof on Linux)
amduprof --session-dir=./uProfSession \
        --collect=timeline,hw-counters \
        -- ./my_app    # run the target application (comment: uProf records samples while app runs)
# after completion, uProf writes files into ./uProfSession; inspect via uProf GUI or parse JSON/CSV exports