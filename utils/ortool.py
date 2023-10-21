import tqdm
from tqdm import tqdm

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
from multiprocessing import Pool


def _solve_cp(nodes, veh_count, veh_capa, veh_speed, late_cost):
    manager = pywrapcp.RoutingIndexManager(nodes.size(0), veh_count, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(from_idx, to_idx):
        src = manager.IndexToNode(from_idx)
        dst = manager.IndexToNode(to_idx)
        return int(nodes[src, :2].sub(nodes[dst, :2]).pow(2).sum().pow(0.5))

    d_cb_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(d_cb_idx)

    def dem_cb(idx):
        j = manager.IndexToNode(idx)
        return int(nodes[j, 2])

    q_cb_idx = routing.RegisterUnaryTransitCallback(dem_cb)
    routing.AddDimensionWithVehicleCapacity(q_cb_idx, 0, [veh_capa for _ in range(veh_count)], True, "Capacity")

    if nodes.size(1) > 3:
        horizon = int(nodes[0, 4])

        def time_cb(from_idx, to_idx):
            src = manager.IndexToNode(from_idx)
            dst = manager.IndexToNode(to_idx)
            return int(nodes[src, 5] + nodes[src, :2].sub(nodes[dst, :2]).pow(2).sum().pow(0.5) / veh_speed)

        t_cb_idx = routing.RegisterTransitCallback(time_cb)
        routing.AddDimension(t_cb_idx, horizon, 2 * horizon, True, "Time")
        t_dim = routing.GetDimensionOrDie("Time")
        for j, (e, l) in enumerate(nodes[1:, 3:5], start=1):
            idx = manager.NodeToIndex(j)
            t_dim.CumulVar(idx).SetMin(int(e))
            t_dim.SetCumulVarSoftUpperBound(idx, int(l), late_cost)
        for i in range(veh_count):
            idx = routing.End(i)
            t_dim.SetCumulVarSoftUpperBound(idx, horizon, late_cost)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    assign = routing.SolveWithParameters(params)

    routes = []
    for i in range(veh_count):
        route = []
        idx = routing.Start(i)
        while not routing.IsEnd(idx):
            idx = assign.Value(routing.NextVar(idx))
            route.append(manager.IndexToNode(idx))
        routes.append(route)

    return routes


def ortool_solve(data, late_cost=1):
    with Pool() as p:
        with tqdm(desc="Calling ORTools", total=data.batch_size) as pbar:
            results = [p.apply_async(_solve_cp, (nodes, data.vehicle_count, data.vehicle_capacity, data.vehicle_speed, late_cost),
                                     callback=lambda _: pbar.update()) for nodes in data.nodes_generate()]
            routes = [res.get() for res in results]
    return routes
