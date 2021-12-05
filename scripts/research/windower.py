from msc.data_utils import get_seiz_onsets, get_preictal_times, get_interictal_times

onsets = get_seiz_onsets(package="surfCO", patient="pat_4000")

print(f"{onsets=}")

preictals = get_preictal_times(package="surfCO", patient="pat_4000")
print(f"{preictals=}")

interictals = get_interictal_times(package="surfCO", patient="pat_4000")

print(f"{interictals=}")