import resolve as rve
import argparse

# parser = argparse.ArgumentParser()
# # parser.add_argument('-j', type=int, default=1)
# parser.add_argument('ms')
# args = parser.parse_args()

for ms in ['./CYG-ALL-2052-2MHZ.ms', './CYG-D-6680-64CH-10S.ms']:
    rve.ms2observations(ms, 'DATA')
rve.ms2observations('./AM754_A030124_flagged.ms', 'DATA', 0)
rve.ms2observations('./AM754_A030124_flagged.ms', 'DATA', 1)
