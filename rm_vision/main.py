from models.detector import generate_targets
from models.geometry import calculate_distance
from models.geometry import is_target_attackable
from models.geometry import get_closest_distance
from models.utils import log
import time
while True:
    coord =  generate_targets()
    distance = calculate_distance(coord)
    closest = get_closest_distance(distance)
    attackable = is_target_attackable(closest)
    log(coord, distance, closest,attackable)
    time.sleep(1)