import os, sys, logging, ray
import gymnasium as gym
import traci, sumolib, itertools, random, time
import traci.constants as tc
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from prettytable import PrettyTable
from ray.rllib.algorithms import ppo
from gymnasium.spaces import Discrete, Box

### Import SUMO Library ###
if 'SUMO_HOME' in os.environ: sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else: sys.exit("please declare environment variable 'SUMO_HOME'")
logger = logging.getLogger(__name__)

### Import SUMO Maps ###
sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui.exe')
#sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo.exe')
sumoCmd = [sumoBinary, "-c", os.path.join("Bari_1","Bari_1.sumocfg")]
sumoNet = sumolib.net.readNet(os.path.join("Bari_1","Bari_1.net.xml"))
sumoAdd = sumolib.net.readNet(os.path.join("Bari_1","Bari_1.add.xml"))

### Initializations ###
trainer_nr = 0
reward_past = -100
baseEdge = "24884043#5"
edges = {baseEdge: 0}
baseRoute = "r_0"
baseRecharge = "cs_0"
battery = 10000 # Wh
battery_reductionRate = 0.135 # Wh/m average consumption per meter % of electric battery according to WLTP (Worldwide Harmonized Light Duty Vehicles Test Procedure)
num_stops = random.randint(6, 6)
# data backup
table = PrettyTable(['ID', 'Final battery', 'Tot En. Cons.',  'Tot En. Charged', 'Tot En. Rigen.', 'Reward', 'Dest.'])
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if not os.path.exists('pics'): os.makedirs('pics')
if not os.path.exists('logs'): os.makedirs('logs')
log_file = os.path.join('logs', f"log_{timestamp}.txt")

bat = []
en_con = []
en_ch = []
en_reg = []
rew = []
rew_tr = []
nr_dest = []
graphic = {
	'Nr. trainer': [],
	'Tot En. Cons.': [],
	'Tot En. Charged': [],
	'Tot En. Reg.': [],
	'Tot En. Cons. Eff.': [],
	'Reward': [],
	'Reward_tr': [],
	'Dest.': []
}

# REWARD AND WEIGHTS
rew_init = 0 #calculate below
rew_dest_opt = 10
rew_dist_opt = 10
rew_dest = 10
rew_insuff_battery = 100
rew_low_battery = 5 #calculate below
rew_return_base = 10 # +/- below
rew_imposs_action = 100
rew_all_dest = 50
rew_max = max(rew_init, rew_dest_opt, rew_dist_opt, rew_dest, rew_insuff_battery,
			  rew_low_battery, rew_return_base, rew_imposs_action, rew_all_dest)

weight_init = 0.5
weight_dest_opt = 5
weight_dist_opt = 2
weight_dest = 3
weight_insuff_battery = 1
weight_low_battery = 1
weight_return_base = 2
weight_imposs_action = 0.5
weight_all_dest = 2
msg_data = f'Weight used: init - {weight_init}, dest_opt - {weight_dest_opt}, dist_opt - {weight_dist_opt}, ' \
           f'dest - {weight_dest}, ins. battery - {weight_insuff_battery}, low battery - {weight_low_battery}, ' \
           f'return base - {weight_return_base}, imp. action - {weight_imposs_action}, all dest. - {weight_all_dest}'
with open(log_file, 'a') as file: file.write(msg_data + '\n')

# CLASS FOR TRAINING
class routeplanner(gym.Env):

	ego_idx = -1
	current_ego = "EGO_0"
	optimalRoute = [baseEdge]
	edges = []
	prev_dist = 0
	energy_charged_past = 0

	def __init__(self, env_config):
		traci.start(sumoCmd)
		self.reward_ep = 0
		self.estimate_init = 0
		self.min_route = []
		self.min_route_casual = []
		self.destination_done = []
		self.check_route_original = []
		self.check_route = []
		self.edges = traci.edge.getIDList()
		stops_list = [edge for edge in self.edges if not edge.startswith(":")]
		stops_list.remove(baseEdge)
		stops_list = ['380400969#1', '30029562#6', '30029559#0', '30029559#2', '24884043#3.3', '68576723#1'] # for test
		self.prev_road = baseEdge
		stops = random.sample(stops_list, num_stops)
		self.add_visuals(stops)
		self.min_distance, self.min_route_original = self.min_distance_stops(stops, baseEdge)
		self.save_log('Number of stops: ' + str(num_stops) + ', optmial combination route (with start and stop from/to base): ' + ', '.join(self.min_route_original))
		self.action_space = Discrete(3)
		self.observation_space = Discrete(len(self.edges))
		self.resetSimulation()

	def reset(self):
		return self.edges.index(baseEdge)

	def step(self, action):
		done = False
		destination = False
		action_applied = False
		return_base = False

		traci.simulationStep()
		ego_values = traci.vehicle.getSubscriptionResults(self.current_ego)
		current_road = ego_values.get(tc.VAR_ROAD_ID, "")
		self.road_current = current_road
		velocity = ego_values.get(tc.VAR_SPEED, "")
		battery_current = float(traci.vehicle.getParameter(self.current_ego, "device.battery.actualBatteryCapacity"))
		energy_charged = traci.vehicle.getParameter(self.current_ego, "device.battery.energyCharged")
		energy_consumption = traci.vehicle.getElectricityConsumption(self.current_ego) # Wh
		distance_travelled = traci.vehicle.getDistance(self.current_ego)  # meters
		self.energy_charged_past = energy_charged
		if energy_consumption > 0: self.energy_consunption_tot += energy_consumption

		reward = 0
# INITIAL REWARD, BASED ON DISTANCE AND ENERGY CONSUMPTION
		if velocity > 0:
			if distance_travelled > self.min_distance*2:
				reward -= (rew_imposs_action / rew_max * weight_imposs_action)*len(self.min_route_original)
				done = True

			if distance_travelled > 0 and self.energy_consunption_tot > 0:
				reward += (distance_travelled / self.energy_consunption_tot) / rew_max * weight_init

# REWARD FOR DESTINATIONS
			if current_road == self.min_route[0]:
				if current_road != baseEdge:
					traci.polygon.setColor(current_road, (0, 255, 0, 255)) # Green
					self.min_route_casual.remove(current_road)
					self.destination_done.append(current_road)
				self.destination_done_nr += 1
				self.distance_travelled_previous = distance_travelled
				destination = True
				reward += rew_dest_opt / rew_max * weight_dest_opt# * self.destination_done_nr
				if distance_travelled - self.dist_prev_dest <= self.dist_next_dest:
					reward += rew_dist_opt / rew_max * weight_dist_opt
				else: 
					reward -= (distance_travelled - self.dist_prev_dest - self.dist_next_dest)/1000 * rew_dist_opt / rew_max * weight_dist_opt
			elif current_road in self.min_route_casual:
				traci.polygon.setColor(current_road, (255, 255, 0, 255)) # Yellow
				self.min_route_casual.remove(current_road)
				self.min_route.remove(current_road)
				self.destination_done.append(current_road)
				self.destination_done_nr += 1
				self.min_distance, self.min_route = self.min_distance_stops(self.min_route, current_road)
				destination = True
				reward += rew_dest / rew_max * weight_dest# * self.destination_done_nr
				self.distance_travelled_previous = distance_travelled

# REWARD (PENALTY) FOR INSUFFICIENT BATTERY
			battery_return = self.est_battery_next_dest(current_road, baseEdge)
			if battery_current < battery_return:
				reward -= rew_insuff_battery / rew_max * weight_insuff_battery
				done = True

# REWARD (PENALTY) FOR INSUFFICIENT BATTERY FOR NEXT DESTINATION
			battery_necessary = self.est_battery_next_dest(current_road, self.min_route[0])
			if battery_current < battery_necessary + battery_return:
				try: current_dist = self.get_dist_next_dest(current_road, baseEdge)
				except: current_dist = 1
				if battery_current < battery_return: reward -= (current_dist / (battery_return - battery_current)) / rew_max * weight_low_battery
				else: reward += rew_low_battery / rew_max * weight_low_battery

# REWARD (OR PENALTY) FOR RETURN TO RECHARGE BASE
			if return_base and energy_charged != self.energy_charged_past:
				reward += rew_return_base / rew_max * weight_return_base
				traci.vehicle.setParameter(self.current_ego, "device.battery.actualBatteryCapacity", battery)
			elif return_base == False and energy_charged != self.energy_charged_past:
				reward -= rew_return_base / rew_max * weight_return_base

# CHOICE ACTION AND REWARD (PENALTY) FOR IMPOSSIBLE ACTION
			if current_road != self.prev_road and action_applied == False:
				outEdges={}
				try: outEdges = sumoNet.getEdge(current_road).getOutgoing()
				except Exception: pass
				outEdgesList = [outEdge.getID() for outEdge in outEdges]
				if len(outEdgesList)>0:
					if action >= len(outEdgesList):
						reward -= (rew_imposs_action / rew_max * weight_imposs_action)*len(self.min_route_original)
						if distance_travelled - self.dist_prev_dest - self.dist_next_dest > 0:
							reward -= (distance_travelled - self.dist_prev_dest - self.dist_next_dest)/1000 * rew_dist_opt / rew_max * weight_dist_opt
						done = True
					else:
						self.optimalRoute.append(outEdgesList[action])
						traci.vehicle.setRoute(self.current_ego,[current_road, outEdgesList[action]])
						action_applied = True
						if current_road == self.check_route[0]:
							reward += rew_dist_opt / rew_max * weight_dist_opt
							self.check_route.pop(0)
						else: reward -= rew_dist_opt / rew_max * weight_dist_opt

# SET NEXT DESTINATION
			if destination and len(self.min_route) > 1:
				self.dist_prev_dest += self.dist_next_dest
				self.dist_next_dest = self.get_dist_next_dest(self.min_route[0], self.min_route[1]) # calculation of distance to next destination
				self.min_route.pop(0) # removed the destination of the delivery just made
			elif destination: self.min_route.pop(0)

# CHECK ALL DESTINATIONS DONE
			if len(self.min_route) == 0: 
				done = True
				reward += rew_all_dest / rew_max * weight_all_dest

# UPDATE MEMORY VARIABLES
			self.prev_road = current_road
			self.reward_ep += reward

# RESET SIMULATION
			if done: self.resetSimulation()
			if self.reward_ep < -20: self.reward_ep = -20
		return self.edges.index(current_road), reward, done, {}

	def resetSimulation(self):
		if (self.ego_idx > -1 and self.current_ego in traci.vehicle.getIDList()):
			battery_final = traci.vehicle.getParameter(self.current_ego, "device.battery.actualBatteryCapacity")
			totEneCon = traci.vehicle.getParameter(self.current_ego, "device.battery.totalEnergyConsumed")
			totEneCha = traci.vehicle.getParameter(self.current_ego, "device.battery.energyCharged")
			totEneReg = traci.vehicle.getParameter(self.current_ego, "device.battery.totalEnergyRegenerated")
			bat.append(float(battery_final))
			en_con.append(float(totEneCon))
			en_ch.append(float(totEneCha))
			en_reg.append(float(totEneReg))
			rew.append(float(self.reward_ep))
			nr_dest.append(float(self.destination_done_nr))
			self.reward_ep = f'{self.reward_ep:.2f}'
			table.add_row([self.current_ego, str(battery_final), str(totEneCon), str(totEneCha), str(totEneReg), str(self.reward_ep), str(self.destination_done_nr)])
			traci.vehicle.unsubscribe(self.current_ego)
			traci.vehicle.remove(self.current_ego)

		self.reward_ep = 0
		self.ego_idx += 1
		self.dist_prev_dest = 0
		self.energy_consunption_tot = 0
		self.current_ego = "EGO_" + str(self.ego_idx)
		self.optimalRoute = [baseEdge]

		traci.vehicle.add(self.current_ego, baseRoute, typeID="electricVehicle", departPos="45")
		traci.vehicle.setParameter(self.current_ego, "device.battery.actualBatteryCapacity", battery)
		traci.vehicle.setParameter(self.current_ego, "device.battery.maximumBatteryCapacity", battery)
		traci.vehicle.subscribe(self.current_ego, (
			tc.VAR_ROUTE_ID,
			tc.VAR_ROAD_ID,
			tc.VAR_POSITION,
			tc.VAR_SPEED,
		))
		self.dist_next_dest = self.get_dist_next_dest(self.min_route_original[0], self.min_route_original[1])

		self.min_route = self.min_route_original[1:]
		self.min_route_casual = self.min_route_original[1:-1]
		try: self.destination_done.remove(baseEdge)
		except: pass
		for dest in self.destination_done: traci.polygon.setColor(dest, (255, 0, 0, 255)) # Rosso
		self.check_route = self.check_route_original[:]
		self.destination_done = []
		self.distance_travelled_previous = 0
		self.destination_done_nr = 0
	
	def add_visuals_route(self, route):
		# visual addition on the map of route
		i = 0
		route.pop(0)
		for stop in route:
			if f'r_{stop}' not in traci.polygon.getIDList():
				i += 1
				stop_position_shape = sumoNet.getEdge(stop).getShape()
				stop_position = []
				stop_position.append((stop_position_shape[0][0] + stop_position_shape[-1][0])/2 + (stop_position_shape[0][0] - stop_position_shape[-1][0])/3)
				stop_position.append((stop_position_shape[0][1] + stop_position_shape[-1][1])/2 + (stop_position_shape[0][1] - stop_position_shape[-1][1])/3)
				traci.polygon.add(
					polygonID = f'r_{stop}',
					shape = [(stop_position[0] - 3.5, stop_position[1] - 3.5), 
						(stop_position[0] - 3.5, stop_position[1] + 3.5), 
						(stop_position[0] + 3.5, stop_position[1] + 3.5), 
						(stop_position[0] + 3.5, stop_position[1] - 3.5)],
					color = (75, 75, 75, 255),  # Gray
					layer = 0,
					fill = True,
					lineWidth = 0.5
            	)

	def add_visuals(self, stops):
		# visual addition on the map of delivery points
		for stop in stops:
			stop_position_shape = sumoNet.getEdge(stop).getShape()
			stop_position = []
			stop_position.append((stop_position_shape[0][0] + stop_position_shape[-1][0])/2)
			stop_position.append((stop_position_shape[0][1] + stop_position_shape[-1][1])/2)
			traci.polygon.add(
                polygonID = stop,
                shape = [(stop_position[0] - 5, stop_position[1] - 5), 
                       (stop_position[0] - 5, stop_position[1] + 5), 
                       (stop_position[0] + 5, stop_position[1] + 5), 
                       (stop_position[0] + 5, stop_position[1] - 5)],
                color = (255, 0, 0, 255),  # Red
                layer = 0,
                fill = True,
                lineWidth = 0.5
            )

	def est_battery_next_dest(self, road1, road2):
		# calculation of estimated battery consumption necessary for the next delivery
		if road1 == road2:
			estimate_battery_consumaption = 0
		else:
			roadsList = []
			tempRoute = traci.simulation.findRoute(road1, road2)
			roadsList_ = tempRoute.edges
			roadsList = [edge for edge in roadsList_ if not edge.startswith(":")]
			estimate_distance = 0

			for road in roadsList: estimate_distance += sumoNet.getEdge(road).getLength()
			estimate_battery_consumaption = estimate_distance * battery_reductionRate
		return estimate_battery_consumaption

	def estimate_consumption(self, min_route_list, start):
		# calculation of the estimated battery consumption of the electric vehicle
		if self.estimate_init == 0:
			total_distance, roadsList = self.tot_distance(min_route_list, start)
			self.check_route_original = roadsList
			self.add_visuals_route(roadsList)
			self.save_log('Complete minimal route of ' + str(len(roadsList)) + ' edges: ' + str(roadsList))

			estimate_battery_consumation = total_distance * battery_reductionRate
			self.save_log(f'Total length of route is {total_distance:.2f} meters')
			self.save_log(f'Estimate battery consumption: {estimate_battery_consumation:.2f} Wh')
			self.estimate_init = 1
			return total_distance, estimate_battery_consumation
		
	def get_dist_next_dest(self, road1, road2):
		#distance calculation for the next delivery
		dist_next_dest = 0
		tempRoute = traci.simulation.findRoute(road1, road2)
		roadsList = tempRoute.edges
		for road in roadsList:
			distance = sumoNet.getEdge(road).getLength()
			dist_next_dest += distance
		return dist_next_dest

	def tot_distance(self, route, start):
		# route calculation
		total_distance = 0
		roadsList = []
		route = [start] + route + [baseEdge]
		i = 0
		while i < len(route[:-1]):
			tempRoute = traci.simulation.findRoute(route[i], route[i+1])
			edge_list_ids = tempRoute.edges
			roadsList += edge_list_ids[:-1]
			i += 1
		roadsList.append(baseEdge)
		for road in roadsList:
			distance = sumoNet.getEdge(road).getLength()
			total_distance += distance
		return total_distance, roadsList

	def min_distance_stops(self, stops, start):
		# calculation of the most effective delivery order
		min_route = []
		min_distance = float('inf')
		combinations = itertools.permutations(stops)
		combinations = list(combinations)
		for combination in combinations:
			combination = list(combination)
			distance, roadList = self.tot_distance(combination, start)
			if distance < min_distance: 
				min_distance = distance
				min_route = [start] + combination + [baseEdge]
		self.estimate_consumption(min_route, start)

		return min_distance, min_route

	def save_log(self, data):
		# saving data to txt file
		with open(log_file, 'a') as file:
			file.write(data + '\n')

### TRAINING AND RESULTS ###
ray.init()
algo = ppo.PPO(env=routeplanner, config = {
	"env_config": {},
    "num_workers": 0#,
	#"num_gpus": 0 # for multi processor
})
plt.figure(figsize=(10, 6))

while True:
	res=algo.train()
	trainer_nr += 1
	print(f'Trainer value nr. {trainer_nr}: {res["episode_reward_mean"]:.2f}')
	reward_past =  res["episode_reward_mean"]

	avg_bat = sum(bat) / len(bat)
	avg_en_con = sum(en_con) / len(en_con)
	avg_en_ch = sum(en_ch) / len(en_ch)
	avg_en_reg = sum(en_reg) / len(en_reg)
	avg_rew = sum(rew) / len(rew)
	avg_dest = sum(nr_dest) / len(nr_dest)
	table.add_row(['Average', f'{avg_bat:.2f}', f'{avg_en_con:.2f}', f'{avg_en_ch:.2f}', f'{avg_en_reg:.2f}', f'{avg_rew:.2f}', f'{avg_dest:.2f}'])
	

	graphic['Nr. trainer'].append(trainer_nr)
	graphic['Tot En. Cons.'].append(avg_en_con / battery * 100)
	graphic['Tot En. Charged'].append(avg_en_ch / battery * 100)
	graphic['Tot En. Reg.'].append(avg_en_reg / battery * 100)
	graphic['Tot En. Cons. Eff.'].append((avg_en_con - avg_en_reg - avg_en_ch) / battery * 100)
	graphic['Reward'].append(avg_rew)
	graphic['Reward_tr'].append(res["episode_reward_mean"])
	graphic['Dest.'].append(avg_dest)

	plt.clf()
	plt.plot(graphic['Nr. trainer'], graphic['Reward'], label='Reward')
	plt.plot(graphic['Nr. trainer'], graphic['Reward_tr'], label='Reward_tr')
	plt.plot(graphic['Nr. trainer'], graphic['Dest.'], label='Dest.')
	plt.xlabel('Nr. trainer')
	plt.ylabel('Values')
	plt.title('Training agent')
	plt.legend()
	plt.grid(True)
	plt.savefig(f"pics/pic_{timestamp}_main.png", dpi=300, bbox_inches='tight')

	plt.clf()
	plt.plot(graphic['Nr. trainer'], graphic['Tot En. Cons.'], label='Tot En. Cons.')
	plt.plot(graphic['Nr. trainer'], graphic['Tot En. Charged'], label='Tot En. Charged')
	plt.plot(graphic['Nr. trainer'], graphic['Tot En. Reg.'], label='Tot En. Reg.')
	plt.plot(graphic['Nr. trainer'], graphic['Tot En. Cons. Eff.'], label='Tot En. Cons. Eff.')
	plt.xlabel('Nr. trainer')
	plt.ylabel('Values')
	plt.title('Training agent')
	plt.legend()
	plt.grid(True)
	plt.savefig(f"pics/pic_{timestamp}_energy.png", dpi=300, bbox_inches='tight')
	
	bat = []
	en_con = []
	en_ch = []
	en_rig = []
	en_reg = []
	rew = []
	nr_dest = []

	with open(log_file, 'a') as file:
		file.write(f'Trainer value nr. {trainer_nr}: {res["episode_reward_mean"]:.2f}' + '\n')
		file.write(str(table) + '\n')
	#if res["episode_reward_mean"] > (num_stops*4+10): break # condition for stop simulation
	table = PrettyTable(['ID', 'Final battery', 'Tot En. Cons.',  'Tot En. Charged', 'Tot En. Rigen.', 'Reward', 'Dest.'])