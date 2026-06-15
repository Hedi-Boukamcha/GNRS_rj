from models.state import *
from conf import * 

# ##################################
# =*= STEP-BY-STEP GNN SIMULATOR =*=
# ##################################
__author__  = "Hedi Boukamcha; Anas Neumann"
__email__   = "hedi.boukamcha.1@ulaval.ca; anas.neumann@polymtl.ca"
__version__ = "2.0.0" 
__license__ = "MIT"

def simulate(previous_state: State, d: Decision, clone: bool=False) -> State:
    state: State      = previous_state.clone() if clone else previous_state
    j: JobState       = state.get_job_by_id(d.job_id)
    o: OperationState = j.operation_states[d.operation_id]
    M: int            = state.M
    L: int            = state.L
    robot             = state.robot
    machine: Machine = o.get_target_machine(state)

    # 1. Search for a possible parralel job that needs to stay on "positioner" (cancel previous unloading actions)
    job_on_pos_to_unload: JobState = None
    forbidden_station: StationState = None
    if d.parallel and o.operation.type == MACHINE_2:
        job_on_pos_to_unload, forbidden_station = cancel_unloading_last_parallel_if_exist(state, j.is_big())

    # 2. Search the possible start time (either load the job or wait for its previous op to finish)
    target_job_ready_time: int = search_start_time(state, j, d, forbidden_station)
    
    # 3. Unload previous job if the target machine ain't free
    previous_job_back_to_station(state, robot, j, machine, M)

    # 4. If M2 not in parallel, unload the job in positioner
    if not d.parallel:
        free_positioner(state, robot, M, current_job=j.id)

    # 5. Move the robot if its not at already at job location
    robot_move_to_job(state, j, o, robot, M)

    # 6. Robot moves job to target machine
    target_job_at_machine_time: int = robot_move_to_machine(j, o, robot, machine, M, job_ready_time=target_job_ready_time)

    # 7. Job needs to be placed on the positioner
    if d.parallel and o.operation.type == MACHINE_1:
            # Vérifier si un POS est déjà en cours pour ce job
        last_m1_event = machine.calendar.get_last_event()
        if last_m1_event and last_m1_event.event_type == POS and last_m1_event.job.id == j.id:
            # POS déjà fait → utiliser son end comme start d'exécution
            time_start_of_execution = last_m1_event.end
        else:
            pos_time: int                = position_job(j, o, robot, machine, target_job_at_machine_time)
            time_start_of_execution: int = pos_time
    else:
        time_start_of_execution: int = target_job_at_machine_time

    # 8. Execute the operation
    parallel: bool        = (d.parallel and o.operation.type == MACHINE_1)
    time_end_of_execution = execute_operation(j, o, robot, machine, parallel, time_start_of_execution)

    # 9. If the operation is the last of the job, we remove the job from the system
    if o.is_last:
        #print(f"  o.is_last=True pour J{j.id+1}, robot.free_at={robot.free_at}")
        robot_move_job_to_station(state, robot, j, o, machine, M)
        #print(f"  après robot_move_job_to_station, robot.free_at={robot.free_at}")
        unloading_time_target = unload(state, j, o, L, unloading_start=robot.free_at)
        #print(f"  après unload, robot.free_at={robot.free_at}")
    else:
        """if not parallel:
            robot_move_job_to_station(state, robot, j, o, machine, M)"""
        unloading_time_target = simulate_station_min_free_at(state.robot, j, o, state.M, state.L, time_end_of_execution)

    # 10. If a parallel job was waiting (to be unloaded) on the positioner, unload it
    unloading_time_pos_job: int = -1
    if job_on_pos_to_unload is not None:
        if not o.is_last:
            if j.current_station is not None and j.location != state.all_stations:
                robot_move_job_to_station(state, robot, j, o, machine, M)

        last_op: OperationState = job_on_pos_to_unload.operation_states[-1]
        robot_move_job_to_station(state, robot, job_on_pos_to_unload, last_op, state.machine1, M)
        unloading_time_pos_job = unload(state, job_on_pos_to_unload, last_op, L, unloading_start=robot.free_at) 

    state.compute_obj_values_and_upper_bounds(unloading_time=max(unloading_time_target, unloading_time_pos_job), current_time=robot.free_at)
    state.decisions.append(d)
    return state

# (1/4) SEARCH START TIME AND LOAD A JOB ###################################################################

def search_start_time(state: State, j: JobState, d: Decision, forbidden_station: StationState) -> int:
    start_time: int = 0

    # D'abord respecter la chronologie du job
    if d.operation_id > 0 and j.calendar.has_events():
        start_time = max(start_time, j.calendar.get_last_event().end)
    o: OperationState = j.operation_states[d.operation_id]
    if o.status == IN_EXECUTION and j.calendar.has_events():
        start_time = max(start_time, j.calendar.get_last_event().end)
    # Ensuite seulement, si le job est hors système, on le recharge
    if j.location is None:
        start_time = search_best_station_and_load_job(
            state,
            j,
            forbidden_station,
            min_start_time=start_time
        )
    return start_time

def search_best_station_and_load_job(state: State, j: JobState, forbidden_station: StationState, min_start_time: int = 0) -> int:
    min_possible_loaded_time: int = -1
    selected_station: StationState = None

    for s in state.all_stations.get_possible_stations(j.is_big()):
        if forbidden_station is None or s.id != forbidden_station.id:
            possible_loading_time = max(test_loading_time(state, s), min_start_time)

            if min_possible_loaded_time < 0 or possible_loading_time < min_possible_loaded_time or (
                selected_station.accept_big and not s.accept_big and possible_loading_time <= (1.015 * min_possible_loaded_time)
            ):
                selected_station = s
                min_possible_loaded_time = possible_loading_time

    prev_unload_time: int = get_loading_time_and_force_unloading_previous(state, j, selected_station)
    prev_unload_time = max(prev_unload_time, min_start_time)

    load_time: int = load_job_into_station(state, j, selected_station, state.L, prev_unload_time)
    return load_time

def load_job_into_station(state: State, job: JobState, station: StationState, L: int, start_loading_time: int):
    start_loading_time = max(start_loading_time, job.job.release_date, station.free_at)
    loaded_time: int   = start_loading_time + L if (station.calendar.has_events() or start_loading_time > 0) else start_loading_time
    station.calendar.add(Event(start=start_loading_time, end=loaded_time, event_type=LOAD, job=job, station=station, source=state.all_stations, dest=state.all_stations))
    job.calendar.add(Event(start=start_loading_time, end=loaded_time, event_type=LOAD, job=job, station=station, source=state.all_stations, dest=state.all_stations))
    job.location        = state.all_stations
    job.status          = IN_SYSTEM
    job.current_station = station
    station.current_job = job
    return loaded_time

def get_loading_time_and_force_unloading_previous(state: State, j: JobState, station: StationState) -> int: 
    if station.current_job == None: # Case 1: station is free
        return max(0, station.free_at)
    else: # Case 2: unload the blocking job!
        current_job: JobState       = station.current_job
        last_op: OperationState     = current_job.get_last_executed_operation()
        if current_job.location.position_type == POS_MACHINE_1:
            robot_move_job_to_station(state, state.robot, current_job, last_op, state.machine1, state.M)
        elif current_job.location.position_type == POS_MACHINE_2:
            robot_move_job_to_station(state, state.robot, current_job, last_op, state.machine2, state.M)
        unloading_start = max(
            station.free_at,
            state.robot.free_at,
            current_job.calendar.events[-1].end
        )

        prev_unload_time: int = unload(
            state,
            current_job,
            last_op,
            state.L,
            unloading_start=unloading_start
        )
        return prev_unload_time

def test_loading_time(state: State, station: StationState) -> int: 
    if station.current_job == None: # Case 1: station is free
        return max(0, station.free_at)
    else: # Case 2: station is not free => what time to unload its job?
        current_job: JobState   = station.current_job
        last_op: OperationState = current_job.get_last_executed_operation()
        time: int               = last_op.end if last_op else max(station.free_at, state.robot.free_at)
        if current_job.location.position_type == POS_MACHINE_1 or current_job.location.position_type == POS_MACHINE_2:
            time +=2* state.M if state.robot.location != current_job.location else state.M
        time += state.L
        return time

# (2/4) CANCEL THE UNLOADING OF LAST PARALLEL MODE B #######################################################

def cancel_unloading_last_parallel_if_exist(state: State, needs_station_2: bool):
    if state.machine1.calendar.len() >= 2:
        previous_last_event: Event = state.machine1.calendar.get(-2)
        operation: OperationState = previous_last_event.operation
        j: JobState = previous_last_event.job
        if previous_last_event.event_type == POS and operation.is_last and (not needs_station_2 or j.current_station.id != STATION_2):
            # Après un cut_time, MOVE+UNLOAD n'existent pas encore → ne pas rollback
            if not j.calendar.has_events() or j.calendar.get_last_event().event_type != UNLOAD:
                return None, None
            # Rollback the job
            j.calendar.events.pop()
            j.calendar.events.pop()
            j.status                      = IN_SYSTEM
            j.location                    = state.machine1
            j.operation_states[-1].status = IN_EXECUTION

            # Rollback the station
            j.current_station.calendar.events.pop()
            j.current_station.calendar.events.pop()
            j.current_station.current_job = j
            
            # Rollback the robot
            e: Event    = state.robot.calendar.events.pop()
            prev: Event = state.robot.calendar.events[-1]
            if prev.event_type == MOVE: # if the robot was not already at machine 1
                e = state.robot.calendar.events.pop()
            state.robot.location    = e.source
            state.robot.current_job = None
            state.robot.free_at     = state.robot.calendar.get(-1).end
            return j, j.current_station
    return None, None

# (3/4) FREE THE TARGET MACHINE IF STILL BUSY ##############################################################

def previous_job_back_to_station(state: State, robot: RobotState, j: JobState, machine: Machine, M: int):
    if machine.calendar.has_events():
        previous_job: JobState = machine.calendar.get(-1).job
        previous_event_op: OperationState = machine.calendar.get(-1).operation

        if previous_event_op is None:
            return

        # Important : récupérer la vraie opération depuis le job
        previous_op: OperationState = previous_job.operation_states[previous_event_op.id]

        if (
            previous_job.id != j.id
            and previous_job.location is not None
            and previous_job.location.position_type == machine.position_type
        ):
            robot_move_job_to_station(
                state,
                robot,
                previous_job,
                previous_op,
                machine,
                M
            )
            move_end = robot.free_at
            if previous_op.end <= move_end:
                previous_op.status = DONE
                previous_op.remaining_time = 0

                # Optionnel mais propre : synchroniser aussi l'opération de l'event
                previous_event_op.status = DONE
                previous_event_op.remaining_time = 0
            if previous_op.is_last and previous_job.status != DONE:
                unload(
                    state,
                    previous_job,
                    previous_op,
                    state.L,
                    unloading_start=move_end
                )
            return move_end
                   
def simulate_station_min_free_at(robot: RobotState, j: JobState, o: OperationState, M: int, L: int, time_end_of_execution: int) -> int:
    simulated_time = time_end_of_execution + M + L
    if robot.location != j.location:
        simulated_time += M
    j.current_station.free_at = simulated_time # min time at which the station could be free
    simulated_time            = simulated_time + M + j.operation_states[o.id + 1].operation.processing_time
    j.end                     = simulated_time # min time at which the job could end
    j.delay                   = max(0, j.end - j.job.due_date)
    return simulated_time

def finalize_after_machine_to_station(state: State, j: JobState, o: OperationState, move_end: int):
    """
    Règle générale :
    Dès qu'un job revient d'une machine vers une station,
    si son opération est terminée, on synchronise l'état.
    Si c'est la dernière opération, on déclenche unload.
    """

    if j is None or o is None:
        return None

    # récupérer la vraie opération du job
    real_op = j.operation_states[o.id]

    # Si l'opération est terminée temporellement
    if real_op.end is not None and real_op.end <= move_end:
        real_op.status = DONE
        real_op.remaining_time = 0

        # synchroniser aussi l'objet opération passé dans l'event
        o.status = DONE
        o.remaining_time = 0

    # éviter double unload
    already_unloaded = any(
        e.event_type == UNLOAD
        and e.job is not None
        and e.job.id == j.id
        and e.operation is not None
        and e.operation.id == real_op.id
        for e in j.calendar.events
    )

    if real_op.is_last and j.status != DONE and not already_unloaded:
        return unload(
            state,
            j,
            real_op,
            state.L,
            unloading_start=move_end
        )

    return None

def robot_move_job_to_station(state: State, robot: RobotState, j: JobState, o: OperationState, machine: Machine, M: int):
    robot_move_to_job(state, j, o, robot, M)

    time = max(o.end, robot.free_at, machine.free_at)

    robot.calendar.add(Event(
        start=time,
        end=time + M,
        event_type=MOVE,
        job=j,
        source=machine,
        dest=state.all_stations,
        operation=o,
        station=j.current_station
    ))

    j.calendar.add(Event(
        start=time,
        end=time + M,
        event_type=MOVE,
        job=j,
        source=machine,
        dest=state.all_stations,
        operation=o,
        station=j.current_station
    ))

    machine.free_at = time
    time += M

    robot.location = state.all_stations
    j.location = state.all_stations
    robot.free_at = time

    finalize_after_machine_to_station(state, j, o, time)

    return time

def free_positioner(state: State, robot: RobotState, M: int, current_job: int):
    if state.machine1.calendar.len() >= 2:
        previous_last_event: Event = state.machine1.calendar.get(-2)
        j: JobState                = previous_last_event.job
        o: OperationState          = previous_last_event.operation
        if current_job != j.id and previous_last_event.event_type == POS and j.location == state.machine1:
            if robot.location != state.machine1:
                robot.calendar.add(Event(start=robot.free_at, end=(robot.free_at + M), event_type=MOVE, job=j, source=robot.location, dest=state.machine1, operation=o, station=j.current_station))
                robot.free_at += M
            time            = max(robot.free_at, state.machine1.free_at, o.end)
            robot.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=state.machine1, dest=state.all_stations, operation=o, station=j.current_station))
            j.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=state.machine1, dest=state.all_stations, operation=o, station=j.current_station))
            time           += M
            robot.location  = state.all_stations
            j.location      = state.all_stations
            robot.free_at   = time

            finalize_after_machine_to_station(state, j, o, time)

def unload(state: State, j: JobState, o: OperationState, L: int, unloading_start: int) ->int:
    already_unloaded = next(
        (
            e for e in j.calendar.events
            if e.event_type == UNLOAD
            and e.operation is not None
            and o is not None
            and e.operation.id == o.id
        ),
        None
    )

    if already_unloaded is not None:
        return already_unloaded.end
    s: StationState      = j.current_station
    unloading_end: int   = unloading_start + L
    j.calendar.add(Event(start=unloading_start, end=unloading_end, event_type=UNLOAD, job=j, source=state.all_stations, dest=state.all_stations, operation=o, station=j.current_station))
    #j.current_station.calendar.add(Event(start=j.current_station.calendar.events[-1].end, end=unloading_start, event_type=AWAIT, job=j, source=state.all_stations, dest=state.all_stations, operation=o, station=j.current_station))
    last_station_event_for_job = next((e for e in reversed(s.calendar.events) if e.job and e.job.id == j.id), None)
    last_station_end = last_station_event_for_job.end if last_station_event_for_job else unloading_start
    if last_station_end < unloading_start:
        j.current_station.calendar.add(Event(
            start=last_station_end,
            end=unloading_start,
            event_type=AWAIT,
            job=j,
            source=state.all_stations,
            dest=state.all_stations,
            operation=o,
            station=j.current_station
        ))
    j.current_station.calendar.add(Event(start=unloading_start, end=unloading_end, event_type=UNLOAD, job=j, source=state.all_stations, dest=state.all_stations, operation=o, station=j.current_station))
    s.free_at          = unloading_end
    s.current_job      = None
    j.end              = unloading_end
    #j.current_station = None
    j.delay            = max(0, j.end - j.job.due_date)
    j.status           = DONE if o and o.is_last else NOT_YET
    j.location         = None
    if o is not None:
        o.status = DONE
        o.remaining_time = 0
    return unloading_end

# (4/4) EXECUTE ONE OPERATION ##############################################################################

def execute_operation(j: JobState, o: OperationState, robot: RobotState, machine: Machine, parallel: bool, time: int) -> int:
    #execution_time: int = o.operation.processing_time
    execution_time: int = o.remaining_time if o.status == IN_EXECUTION else o.operation.processing_time  # ← MODIFIER ICI
    o.start             = time
    if not parallel:
        robot.free_at   = time + execution_time
        robot.calendar.add(Event(start=time, end=(time + execution_time), event_type=HOLD, job=j, source=machine, dest=machine, operation=o, station=None))
    j.calendar.add(Event(start=time, end=(time + execution_time), event_type=EXECUTE, job=j, source=machine, dest=machine, operation=o, station=None))
    machine.calendar.add(Event(start=time, end=(time + execution_time), event_type=EXECUTE, job=j, source=machine, dest=machine, operation=o, station=None))
    time               += execution_time
    machine.free_at     = time
    o.end               = time
    o.status            = DONE
    o.remaining_time    = 0
    return time

def robot_move_to_job(state: State, j: JobState, o: OperationState, robot: RobotState, M: int):
    if robot.calendar.has_events():
        last_event = robot.calendar.get_last_event()

        if (
            last_event.event_type == HOLD
            and last_event.job is not None
            and last_event.job.id != j.id
        ):
            held_job = last_event.job
            held_op = last_event.operation
            held_machine = last_event.dest

            # Le robot ne peut pas partir vers j tant qu'il tient held_job.
            robot_move_job_to_station(
                state,
                robot,
                held_job,
                held_op,
                held_machine,
                M
            )
    if robot.location  != j.location:
        # Vérifier si le robot est déjà en route vers ce job
        if robot.calendar.has_events() and robot.calendar.get_last_event().dest == j.location:
            robot.location = j.location
            return
        s: StationState = j.current_station if j.location.position_type == POS_STATION else None
        robot.calendar.add(Event(start=robot.free_at, end=(robot.free_at + M), event_type=MOVE, job=j, source=robot.location, dest=j.location, operation=None, station=s))
        robot.location  = j.location
        robot.free_at  += M

def robot_move_to_machine(j: JobState, o: OperationState, robot: RobotState, machine: Machine, M: int, job_ready_time: int) -> int:
    time = max(job_ready_time, robot.free_at, machine.free_at)

    if j.status == DONE or j.location is None:
        print(
            f"WARNING robot_move_to_machine ignored for J{j.id+1}:",
            "status =", j.status,
            "location =", j.location,
            "operation =", o.id + 1
        )
        return time

    if robot.location != machine:
        s: StationState = j.current_station if j.location.position_type == POS_STATION else None

        robot.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=robot.location, dest=machine, operation=o, station=s))
        j.calendar.add(Event(start=time, end=(time + M), event_type=MOVE, job=j, source=robot.location, dest=machine, operation=o, station=s))

        robot.location = machine
        j.location = machine
        time += M
        robot.free_at = time

    else:
        robot.location = machine

        # Seulement si le job est physiquement déjà avec le robot / encore actif
        if j.status != DONE and j.location is not None:
            j.location = machine

    return time

def position_job(j: JobState, o: OperationState, robot: RobotState, machine: Machine, time: int) -> int:
    j.calendar.add(Event(start=time, end=(time + j.job.pos_time), event_type=POS, job=j, source=machine, dest=machine, operation=o, station=None))
    machine.calendar.add(Event(start=time, end=(time + j.job.pos_time), event_type=POS, job=j, source=machine, dest=machine, operation=o, station=None))
    robot.calendar.add(Event(start=time, end=(time + j.job.pos_time), event_type=POS, job=j, source=machine, dest=machine, operation=o, station=None))
    time           += j.job.pos_time
    robot.free_at   = time
    return time

# SIMULATOR WITH CUT TIME ##########################################################################################

def _cut_filter(events, cut_time, keep_in_progress=True):
    """Filtre les événements selon cut_time."""
    if keep_in_progress:
        return [e for e in events if e.end <= cut_time or (e.start <= cut_time and e.end > cut_time)]
    return [e for e in events if e.end <= cut_time]

def _cut_robot(new_state: State, cut_time: int):
    new_state.robot.calendar.events = [e for e in new_state.robot.calendar.events if e.end <= cut_time or (e.start < cut_time and e.end > cut_time)]
    new_state.robot.free_at  = new_state.robot.calendar.events[-1].end if new_state.robot.calendar.events else 0
    new_state.robot.location = new_state.robot.calendar.events[-1].dest if new_state.robot.calendar.events else new_state.all_stations
    #print(f"Robot après cut: free_at={new_state.robot.free_at}, location={new_state.robot.location.position_type}")
    #print(f"Dernier event robot: {new_state.robot.calendar.events[-1]}")
    #print(f"Events avant filtre:")
    #for e in new_state.robot.calendar.events:
        #print(f"  start={e.start}, end={e.end}")
    #print(f"  Filtre robot: cut_time={cut_time}")
    #for e in new_state.robot.calendar.events:
        #garde = e.end <= cut_time or (e.start < cut_time and e.end > cut_time)
        #print(f"    start={e.start}, end={e.end}, gardé={garde}")
    #print(f"Events après filtre: {len(new_state.robot.calendar.events)}")
    #after = len(new_state.robot.calendar.events)
    #print(f"  Robot: {before} events → {after} events après filtre (cut={cut_time})")
    # Mettre à jour current_job du robot
    #ast_hold = next((e for e in reversed(new_state.robot.calendar.events) if e.event_type == HOLD), None)
    #if last_hold and last_hold.end >= cut_time:
        #held_job = new_state.get_job_by_id(last_hold.job.id)
        #print(f"\n=== Robot tient J{held_job.id+1} à cut={cut_time} ===")
        #print(f"  Calendrier J{held_job.id+1}:")
        #for e in held_job.calendar.events:
            #print(f"    start={e.start}, end={e.end}, type={EVENT_NAMES[e.event_type]}, op={e.operation.id if e.operation else None}")

def _fix_robot_held_job(new_state: State, cut_time: int):
    last_hold = next((e for e in reversed(new_state.robot.calendar.events) if e.event_type == HOLD), None)
    if last_hold and last_hold.end > cut_time:
        held_job = new_state.get_job_by_id(last_hold.job.id)
        new_state.robot.current_job = held_job
        last_op = held_job.get_last_executed_operation()
        print(f"  _fix_robot: J{held_job.id+1}, last_op={last_op}, is_last={last_op.is_last if last_op else None}")
        if last_op and last_op.is_last:
            # Le robot ramène le job à la station puis décharge : MOVE légitime
            return_time = new_state.robot.free_at + new_state.M
            new_state.robot.calendar.events.append(Event(
                start=new_state.robot.free_at, end=return_time,
                event_type=MOVE, job=held_job, source=new_state.robot.location,
                dest=new_state.all_stations, operation=last_hold.operation, station=held_job.current_station
            ))
            held_job.calendar.events.append(Event(
                start=new_state.robot.free_at, end=return_time,
                event_type=MOVE, job=held_job, source=new_state.robot.location,
                dest=new_state.all_stations, operation=last_hold.operation, station=held_job.current_station
            ))
            new_state.robot.free_at  = return_time
            new_state.robot.location = new_state.all_stations
            unload_end   = return_time + new_state.L
            held_station = held_job.current_station

            # Ajouter l'AWAIT avant l'UNLOAD dans la station
            if held_station and held_station.calendar.has_events():
                last_station_end = held_station.calendar.events[-1].end
                if last_station_end < return_time:
                    held_station.calendar.events.append(Event(
                        start=last_station_end, end=return_time,
                        event_type=AWAIT, job=held_job, source=new_state.all_stations,
                        dest=new_state.all_stations, operation=last_op, station=held_station
                    ))

            # Deux objets Event distincts (pas le même partagé)
            held_job.calendar.events.append(Event(
                start=return_time, end=unload_end,
                event_type=UNLOAD, job=held_job, source=new_state.all_stations,
                dest=new_state.all_stations, operation=last_op, station=held_station
            ))
            if held_station:
                held_station.calendar.events.append(Event(
                    start=return_time, end=unload_end,
                    event_type=UNLOAD, job=held_job, source=new_state.all_stations,
                    dest=new_state.all_stations, operation=last_op, station=held_station
                ))
                held_station.free_at     = unload_end
                held_station.current_job = None

            held_job.status   = DONE
            held_job.end      = unload_end
            held_job.delay    = max(0, unload_end - held_job.job.due_date)
            held_job.location = None

            if last_op:
                last_op.status = DONE
                last_op.remaining_time = 0

            # Le robot n'intervient pas dans l'unload.
            # Il est libre dès qu'il a ramené le job à la station.
            new_state.robot.free_at = return_time
            new_state.robot.location = new_state.all_stations

        new_state.robot.current_job = None
    else:
        new_state.robot.current_job = None

def _cut_machine(machine, cut_time: int):
    machine.calendar.events = [e for e in machine.calendar.events 
                                if e.end <= cut_time or (e.start <= cut_time and e.end > cut_time)]
    machine.free_at = machine.calendar.events[-1].end if machine.calendar.events else 0

"""def _cut_station(station, new_state: State, cut_time: int):
    #station.calendar.events = _cut_filter(station.calendar.events, cut_time, keep_in_progress=False)
    station.calendar.events = [e for e in station.calendar.events 
                                if e.end <= cut_time or 
                                (e.start <= cut_time and e.end > cut_time and e.event_type in {AWAIT, UNLOAD})]
    station.free_at         = station.calendar.events[-1].end if station.calendar.events else 0
    last_load               = next((e for e in reversed(station.calendar.events) if e.event_type == LOAD), None)
    last_unload             = next((e for e in reversed(station.calendar.events) if e.event_type == UNLOAD), None)
    if last_load and (last_unload is None or last_load.end > last_unload.end):
        station.current_job = new_state.get_job_by_id(last_load.job.id)
    else:
        station.current_job = None"""

def _cut_station(station, new_state: State, cut_time: int):
    new_events = []
    for e in station.calendar.events:
        if e.end <= cut_time:
            new_events.append(e)
        elif e.start <= cut_time < e.end and e.event_type == AWAIT:
            # on coupe l'attente au cut_time, on ne garde pas le futur
            new_events.append(Event(
                start=e.start,
                end=cut_time,
                event_type=e.event_type,
                job=e.job,
                source=e.source,
                dest=e.dest,
                operation=e.operation,
                station=e.station
            ))
    station.calendar.events = new_events
    station.free_at = station.calendar.events[-1].end if station.calendar.events else 0
    last_load = next((e for e in reversed(station.calendar.events) if e.event_type == LOAD), None)
    last_unload = next((e for e in reversed(station.calendar.events) if e.event_type == UNLOAD), None)
    if last_load and (last_unload is None or last_load.end > last_unload.end):
        station.current_job = new_state.get_job_by_id(last_load.job.id)
    else:
        station.current_job = None
    if station.current_job is not None:
        station.free_at = max(station.free_at, cut_time)

def _cut_job_not_yet(j: JobState):
    """Job sans aucun event → NOT_YET."""
    j.status          = NOT_YET
    j.location        = None
    j.current_station = None
    for o in j.operation_states:
        o.status         = NOT_YET
        o.remaining_time = o.operation.processing_time
        o.start          = 0
        o.end            = 0

def _cut_job_done(j: JobState, last_event):
    """Job complètement déchargé → DONE."""
    j.status   = DONE
    j.end      = last_event.end
    j.delay    = max(0, j.end - j.job.due_date)
    j.location = None
    for o in j.operation_states:
        o.remaining_time = 0
        o.status         = DONE

def _cut_job_in_execution(j: JobState, in_progress_event, new_state: State, cut_time: int):
    """Event en cours (EXECUTE/HOLD/POS/MOVE) → IN_EXECUTION."""
    #print(f"  _cut_job_in_execution: J{j.id+1}, event={EVENT_NAMES[in_progress_event.event_type]}, op={in_progress_event.operation.id if in_progress_event.operation else None}, end={in_progress_event.end}")
    cloned_op = j.get_operation(in_progress_event.operation.id) if in_progress_event.operation else None
    #print(f"  cloned_op={cloned_op}, status avant={cloned_op.status if cloned_op else None}")
    j.status   = IN_EXECUTION
    dest = in_progress_event.dest
    if dest is not None:
        if dest.position_type == POS_MACHINE_1:
            j.location = new_state.machine1
        elif dest.position_type == POS_MACHINE_2:
            j.location = new_state.machine2
        else:
            j.location = new_state.all_stations
    if in_progress_event.event_type == MOVE:
        new_state.robot.location = in_progress_event.dest
        j.status = IN_SYSTEM
    original_op = in_progress_event.operation
    if original_op is not None and in_progress_event.event_type in {EXECUTE, HOLD}:
        cloned_op = j.get_operation(original_op.id)
        if cloned_op is not None:
            cloned_op.status         = IN_EXECUTION
            cloned_op.remaining_time = max(0, in_progress_event.end - cut_time)
            for o in j.operation_states:
                if o.id < cloned_op.id:
                    o.remaining_time = 0
                    o.status         = DONE
                elif o.id > cloned_op.id:
                    o.remaining_time = o.operation.processing_time
                    o.status         = NOT_YET

    elif original_op is not None and in_progress_event.event_type == POS:
        cloned_op = j.get_operation(original_op.id)
        if cloned_op is not None:
            cloned_op.status         = NOT_YET
            cloned_op.remaining_time = cloned_op.operation.processing_time
            j.status                 = IN_SYSTEM
            for o in j.operation_states:
                if o.id < cloned_op.id:
                    o.remaining_time = 0
                    o.status         = DONE
                elif o.id > cloned_op.id:
                    o.remaining_time = o.operation.processing_time
                    o.status         = NOT_YET

    else:
        # MOVE en cours → ops selon leur état réel
        for o in j.operation_states:
            if o.start > 0 and o.end <= cut_time:
                o.remaining_time = 0
                o.status         = DONE
            else:
                o.remaining_time = o.operation.processing_time
                o.status         = NOT_YET

def _cut_job_in_system(j: JobState, last_event, cut_time: int):
    """Job dans le système mais pas en exécution → IN_SYSTEM."""
    j.status   = IN_SYSTEM
    j.location = last_event.dest
    for o in j.operation_states:
        if o.start > 0 and o.end <= cut_time:
            o.remaining_time = 0
            o.status         = DONE
        else:
            o.remaining_time = o.operation.processing_time
            o.status         = NOT_YET

def _cut_job(j: JobState, original_job: JobState, new_state: State, cut_time: int):
    """Reconstruit l'état d'un job au cut_time."""
    j.calendar.events     = _cut_filter(j.calendar.events, cut_time, keep_in_progress=True)
    in_progress_event     = next((e for e in original_job.calendar.events if e.start <= cut_time and e.end > cut_time), None)

    if not j.calendar.events and in_progress_event is None:
        _cut_job_not_yet(j)
    elif in_progress_event is not None:
        _cut_job_in_execution(j, in_progress_event, new_state, cut_time)
    else:
        last_event = j.calendar.events[-1]
        if last_event.event_type == UNLOAD:
            _cut_job_done(j, last_event)
        else:
            # vérifier si toutes les ops sont DONE
            all_done = all(o.end > 0 and o.end <= cut_time for o in j.operation_states)
            if all_done:
                _cut_job_done(j, last_event)
            else:
                _cut_job_in_system(j, last_event, cut_time)
    #print(f"J{j.id+1} events après filtre: {len(j.calendar.events)}")

def _clean_robot_obsolete_events(new_state: State, cut_time: int):
    """Garder seulement les events terminés + en cours + move de retour ajouté."""
    new_state.robot.calendar.events = [
        e for e in new_state.robot.calendar.events
        if e.end <= cut_time or (e.start < cut_time and e.end > cut_time) or e.start == new_state.robot.free_at - new_state.M
    ]
    if new_state.robot.calendar.events:
        new_state.robot.free_at  = new_state.robot.calendar.events[-1].end
        new_state.robot.location = new_state.robot.calendar.events[-1].dest



def build_state_from_cut(state: State, cut_time: int) -> State:
    new_state: State = state.clone()
    #print(f"id(new_state.robot.calendar) = {id(new_state.robot.calendar)}")
    #print(f"id(state.robot.calendar) = {id(state.robot.calendar)}")
    _cut_robot(new_state, cut_time)
    for machine in [new_state.machine1, new_state.machine2]:
        _cut_machine(machine, cut_time)
    for station in new_state.all_stations.stations:
        _cut_station(station, new_state, cut_time)
    for j in new_state.job_states:
        _cut_job(j, state.get_job_by_id(j.id), new_state, cut_time)
    
    _clean_robot_obsolete_events(new_state, cut_time)  # 1. nettoyer d'abord
    _fix_robot_held_job(new_state, cut_time)            # 2. ajouter les events post-HOLD après
    return new_state

# END OF FILE! ##########################################################################################