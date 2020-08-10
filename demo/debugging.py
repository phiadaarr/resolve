import resolve as rve


def save_and_load_hdf5(obs):
    for ob in obs:
        ob.save_to_hdf5('foo.hdf5')
        ob1 = rve.Observation.load_from_hdf5('foo.hdf5')
        assert ob == ob1


def main():
    for ms in ['./CYG-ALL-2052-2MHZ.ms', './CYG-D-6680-64CH-10S.ms']:
        obs = rve.ms2observations(ms, 'DATA')
        save_and_load_hdf5(obs)
    obs = rve.ms2observations('./AM754_A030124_flagged.ms', 'DATA', 0)
    save_and_load_hdf5(obs)
    obs = rve.ms2observations('./AM754_A030124_flagged.ms', 'DATA', 1)
    save_and_load_hdf5(obs)


if __name__ == '__main__':
    main()
