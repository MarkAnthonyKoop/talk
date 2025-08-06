#!/bin/bash
# Wrapper script to run Talk with correct Python environment

# Use Python 3.11 which has pydantic-settings installed
export PYTHONPATH="/home/xx/.local/lib/python3.11/site-packages:$PYTHONPATH"

# Run the requested Talk script
if [ "$1" == "v3" ]; then
    shift
    exec python3 /home/xx/code/talk/talk_v3_planning.py "$@"
elif [ "$1" == "v4" ]; then
    shift
    exec python3 /home/xx/code/talk/talk_v4_validated.py "$@"
else
    exec python3 /home/xx/code/talk/talk.py "$@"
fi