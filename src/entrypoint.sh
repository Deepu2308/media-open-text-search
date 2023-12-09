#!/bin/sh

echo Testing by running test.py

# Run the test.py script
python test.py

# Start the Jupyter Notebook server
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root