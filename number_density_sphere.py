from shared.preface import *
import shared.functions as fct

total_start = time.perf_counter()












total_time = time.perf_counter()-total_start
print(f'Total time: {total_time/60.} min, {total_time/(60**2)} h.')