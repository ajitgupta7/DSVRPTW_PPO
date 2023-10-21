import torch


class DSVRPTW_Environment:
    # vehicle coordinates(x_i,y_i), veh_time_time_budget, total_travel_time
    vehicle_feature = 4
    # customers feature: coordinates (x_i,y_i), service time, arrival_time
    customer_feature = 7

    def __init__(self,
                 data=None,
                 nodes=None,
                 vehicle_count=2,
                 vehicle_capacity=200,
                 vehicle_speed=1,
                 pending_cost=2,
                 late_cost=1,
                 speed_var=0.1,
                 late_prop=0.05,
                 slow_down=0.5,
                 late_var=0.2):

        self.vehicle_count = data.vehicle_count if data is not None else vehicle_count
        self.vehicle_capacity = data.vehicle_capacity if data is not None else vehicle_capacity
        self.vehicle_speed = data.vehicle_speed if data is not None else vehicle_speed

        self.nodes = data.nodes if nodes is None else nodes
        self.minibatch, self.nodes_count, _ = self.nodes.size()
        self.pending_cost = pending_cost
        self.late_cost = late_cost

        self.speed_var = speed_var
        self.late_prop = late_prop
        self.slow_down = slow_down
        self.late_var = late_var

    def _sample_speed(self):
        late = self.nodes.new_empty((self.minibatch, 1)).bernoulli_(self.late_prop)
        rand = torch.randn_like(late)
        speed = late * self.slow_down * (1 + self.late_var * rand) + (1 - late) * (1 + self.speed_var * rand)
        return speed.clamp_(min=0.1) * self.vehicle_speed

    def _update_current_vehicles(self, dest, customer_index):
        dist = torch.pairwise_distance(self.current_vehicle[:, 0, :2], dest[:, 0, :2], keepdim=True)
        tt = dist / self._sample_speed()
        arv = torch.max(self.current_vehicle[:, :, 3] + tt, dest[:, :, 3])
        late = (arv - dest[:, :, 4]).clamp_(min=0)

        self.current_vehicle[:, :, :2] = dest[:, :, :2]
        self.current_vehicle[:, :, 2] -= dest[:, :, 2]
        self.current_vehicle[:, :, 3] = arv + dest[:, :, 5]

        self.vehicles = self.vehicles.scatter(1,
                                              self.current_vehicle_index[:, :, None].expand(-1, -1, self.vehicle_feature),
                                              self.current_vehicle)
        return dist, late

    def _done(self, customer_index):
        self.vehicle_done.scatter_(1,
                                   self.current_vehicle_index,
                                   customer_index == 0)

        self.done = bool(self.vehicle_done.all())

    def _update_mask(self, customer_index):
        self.new_customers = False
        self.served.scatter_(1, customer_index, customer_index > 0)
        overload = torch.zeros_like(self.mask).scatter_(1,
                                                        self.current_vehicle_index[:, :, None].expand(-1, -1, self.nodes_count),
                                                        self.current_vehicle[:, :, None, 2] - self.nodes[:, None, :, 2] < 0)

        self.mask = self.mask | self.served[:, None, :] | overload | self.vehicle_done[:, :, None]
        self.mask[:, :, 0] = 0

    # updating current vehicle to find the next available vehicle
    def _update_next_vehicle(self, veh_index=None):
        if veh_index is None:
            avail = self.vehicles[:, :, 3].clone()
            avail[self.vehicle_done] = float('inf')
            self.current_vehicle_index = avail.argmin(1, keepdim=True)
        else:
            self.current_vehicle_index = veh_index

        self.current_vehicle = self.vehicles.gather(1, self.current_vehicle_index[:, :, None].expand(-1, -1,
                                                                                                     self.vehicle_feature))
        self.current_vehicle_mask = self.mask.gather(1, self.current_vehicle_index[:, :, None].expand(-1, -1,
                                                                                                      self.nodes_count))

    def _update_dynamic_customers(self, veh_index):
        time = self.current_vehicle[:, :, 3].clone()
        reveal_dyn_reqs = torch.logical_and((self.customer_mask), (self.nodes[:, :, 6] <= time))
        if reveal_dyn_reqs.any():
            self.new_customer = True
            self.customer_mask = self.customer_mask ^ reveal_dyn_reqs
            self.mask = self.mask ^ reveal_dyn_reqs[:, None, :].expand(-1, self.vehicle_count, -1)
            self.vehicle_done = torch.logical_and(self.vehicle_done, (reveal_dyn_reqs.any(1) ^ True).unsqueeze(1))

            # avail vehicle only when time budget is left
            time_violate = (self.vehicles[:, :, 2] <= 0)
            self.vehicle_done = torch.logical_or(self.vehicle_done, time_violate)

            self.vehicles[:, :, 3] = torch.max(self.vehicles[:, :, 3], time)
            self._update_next_vehicle(veh_index)

    def reset(self):
        # reset vehicle (minibatch*veh_count*veh_feature)
        self.vehicles = self.nodes.new_zeros((self.minibatch, self.vehicle_count, self.vehicle_feature))
        self.vehicles[:, :, :2] = self.nodes[:, :1, :2]
        self.vehicles[:, :, 2] = self.vehicle_capacity

        # reset vehicle done
        self.vehicle_done = self.nodes.new_zeros((self.minibatch, self.vehicle_count), dtype=torch.bool)
        self.done = False

        # initialize reward as tour length
        self.tour_length = torch.zeros((self.minibatch, 1)).to(self.nodes.device)

        # reset cust_mask
        self.customer_mask = self.nodes[:, :, 6] > 0

        # reset new customers and served customer since now to zero (all false)
        self.new_customer = True
        self.served = torch.zeros_like(self.customer_mask)
        self.pending_customers = (self.served ^ True).float().sum(-1, keepdim=True) - 1

        # reset mask (minibatch*veh_count*nodes)
        self.mask = self.customer_mask[:, None, :].repeat(1, self.vehicle_count, 1)

        # reset current vehicle index, current vehicle, current vehicle mask
        self.current_vehicle_index = self.nodes.new_zeros((self.minibatch, 1), dtype=torch.int64)
        self.current_vehicle = self.vehicles.gather(1,
                                                    self.current_vehicle_index[:, :, None].
                                                    expand(-1, -1, self.vehicle_feature))
        self.current_vehicle_mask = self.mask.gather(1,
                                                     self.current_vehicle_index[:, :, None].
                                                     expand(-1, -1, self.nodes_count))

    def get_reward(self, dist, late):
        reward = +dist + self.late_cost * late
        if self.done:
            # penalty for all and static pending customers
            pending_customers = torch.logical_and((self.served ^ True),
                                                  (self.nodes[:, :, 6] >= 0)).float().sum(-1, keepdim=True) - 1
            reward += self.pending_cost * pending_customers
            return reward
        else:
            return reward

    def get_reward1(self, dist, late):
        reward = torch.full_like(dist, -self.pending_cost) + self.late_cost * late
        if self.done:
            # penalty for all and static pending customers
            pending_customers = torch.logical_and((self.served ^ True),
                                                  (self.nodes[:, :, 6] >= 0)).float().sum(-1, keepdim=True) - 1
            reward += self.pending_cost * pending_customers
            reward += self.tour_length
            return reward
        else:
            return reward

    def step(self, customer_index, veh_index=None):
        dest = self.nodes.gather(1, customer_index[:, :, None].expand(-1, -1, self.customer_feature))
        dist, late = self._update_current_vehicles(dest, customer_index)
        self._done(customer_index)
        self._update_mask(customer_index)
        self._update_next_vehicle(veh_index)
        self._update_dynamic_customers(veh_index)

        self.tour_length += dist

        return self.get_reward(dist, late)

    def state_dict(self, dest_dict=None):
        if dest_dict is None:
            dest_dict = {'vehicles': self.vehicles,
                         'vehicle_done': self.vehicle_done,
                         'served': self.served,
                         'mask': self.mask,
                         'current_vehicle_index': self.current_vehicle_index}
        else:
            dest_dict["vehicles"].copy_(self.vehicles)
            dest_dict["vehicle_done"].copy_(self.vehicle_done)
            dest_dict["served"].copy_(self.served)
            dest_dict["mask"].copy_(self.mask)
            dest_dict["current_vehicle_index"].copy_(self.current_vehicle_index)

        return dest_dict

    def load_state_dict(self, state_dict):
        self.vehicles.copy_(state_dict["vehicles"])
        self.vehicle_done.copy_(state_dict["vehicle_done"])
        self.served.copy_(state_dict["served"])
        self.mask.copy_(state_dict["mask"])
        self.current_vehicle_index.copy_(state_dict["current_vehicle_index"])
        self.current_vehicle = self.vehicles.gather(1,
                                                    self.current_vehicle_index[:, :, None].
                                                    expand(-1, -1, self.vehicle_feature))
        self.current_vehicle_mask = self.mask.gather(1, self.current_vehicle_index[:, :, None].
                                                     expand(-1, -1, self.customer_feature))
