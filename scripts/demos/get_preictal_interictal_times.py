from msc.data_utils import get_seiz_onsets, get_preictal_times, get_interictal_times

package = "surfCO"
patient = "pat_3700"

onsets = get_seiz_onsets(package=package, patient=patient)

print(f"{onsets=}")

preictals = get_preictal_times(package=package, patient=patient)
print(f"{preictals=}")

interictals = get_interictal_times(package=package, patient=patient)

print(f"{interictals=}")