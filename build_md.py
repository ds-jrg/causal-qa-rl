#!/usr/bin/env python

import pandas as pd
import sys

with open(sys.argv[1]) as f:
    lines = [line.strip() for line in f]
    table = {name: eval(measures) for name, measures in zip(lines[::2], lines[1::2])}

df = pd.DataFrame(table)

df.T.to_markdown(sys.argv[2])
