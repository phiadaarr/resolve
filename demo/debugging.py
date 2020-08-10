import resolve as rve


def main():
    for ms in ['./CYG-ALL-2052-2MHZ.ms', './CYG-D-6680-64CH-10S.ms']:
        rve.ms2observations(ms, 'DATA')
    rve.ms2observations('./AM754_A030124_flagged.ms', 'DATA', 0)
    rve.ms2observations('./AM754_A030124_flagged.ms', 'DATA', 1)


if __name__ == '__main__':
    main()
