import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from VRP_heuristics import *
import GraphTool


class VNS():
    def __init__(self, graph, iter_num):
        """ 
        read data and preprocess 
        """
        self.graph = graph
        self.iter_num = iter_num
        self.choose_neighbour_strategy = "last"

        # set VNS paraments
        self.operators_list = [Reverse(), Relocate(), Exchange()]
        
    def reset(self):
        self.process = [] # record data while alg running

    def solution_init(self, strategy="heuristic"):
        '''
        generate initial solution (routes), applying VRP_heuristics
        '''
        if strategy == "heuristic":
            alg = Solomon_Insertion(self.graph)
            routes = alg.run()
            # routes = nearest_neighbour(graph)
            solution = []
            for ri, route in enumerate(routes):
                solution.append(0)
                solution += route[1:-1]
            solution.append(0) # add the end 0
        elif strategy == "random":
            solution = list(range(1, self.graph.nodeNum)) + [0]*(7-1) # 7 vehicles indefault
            solution.shuffle()
            solution = [0] + solution + [0]
        return solution

    def transfer(self, solution):
        """
        transfer solution to routes
        """
        routes = []
        for i, p in enumerate(solution[:-1]): # pass the end 0
            if p == 0:
                if i > 0:
                    routes[-1].append(0) # add end 0
                routes.append([0]) # add start 0
            else:
                routes[-1].append(p)
        else:
            routes[-1].append(0) # add final 0
        return routes

    def cal_objective(self, solution):
        '''
        calculate objective of solution (including consideration of soft/hard constraint)
        '''
        obj = 0
        load = 0
        cur_time = 0
        for i in range(1, len(solution)):
            # consideration of distance
            ri = solution[i-1]
            rj = solution[i]
            distance = self.graph.disMatrix[ri, rj]
            obj += distance

            # consideration of graph.capacity
            load += self.graph.demand[ri]
            if load > self.graph.capacity: # break the graph.capacity constraint
                obj += 1000

            # consideration of time window
            cur_time += self.graph.serviceTime[ri] + self.graph.disMatrix[ri, rj]
            cur_time = max(cur_time, self.graph.readyTime[rj]) # if arrived early, wait until ready
            if cur_time > self.graph.dueTime[rj]: # break the TW constraint
                obj += 1000

            # update when back to depot
            if solution[i] == 0:
                load = 0
                cur_time = 0
        return obj
    
    def get_neighbours(self, solution, operator=Relocate()):
        neighbours =  operator.run(solution)
        return neighbours

    def choose_neighbour(self, neighbours):
        # randomly choose neighbour
        if self.choose_neighbour_strategy == "random":
            chosen_ni = np.random.randint(len(neighbours))
        # choose the first neighbour
        elif self.choose_neighbour_strategy == "first":
            chosen_ni = 0
        # choose the first neighbour
        elif self.choose_neighbour_strategy == "last":
            chosen_ni = len(neighbours)-1
        # choose the best neighour
        elif self.choose_neighbour_strategy == "best":
            best_obj = np.inf
            for ni, neighbour in enumerate(neighbours):
                obj = self.cal_objective(neighbour) 
                if obj < best_obj:
                    best_obj = obj
                    best_ni = ni
            chosen_ni = best_ni
        
        return chosen_ni

    def draw(self, routes):
        graph.location = self.graph.location
        plt.scatter(graph.location[:, 0], graph.location[:, 1])
        for route in routes:
            # add depot 0
            x = list(graph.location[route, 0])
            x.append(graph.location[route[0], 0])
            y = list(graph.location[route, 1])
            y.append(graph.location[route[0], 1])
            plt.plot(x, y)
        plt.show()

    def show_process(self):
        y = self.process
        x = np.arange(len(y))
        plt.plot(x, y)
        plt.show()
   
    def run(self):
        self.reset()
        best_solution = self.solution_init() # solution in form of routes
        best_obj = self.cal_objective(best_solution)
        neighbours = self.get_neighbours(best_solution, operator=self.operators_list[0])
        operator_k = 0
        for step in trange(self.iter_num):
            ni = self.choose_neighbour(neighbours)
            cur_solution = neighbours[ni]
            cur_obj = self.cal_objective(cur_solution)
            # obj: minimize the total distance 
            if cur_obj < best_obj: 
                # self.operators_list.insert(0, self.operators_list.pop(operator_k))
                operator_k = 0
                best_solution = cur_solution
                best_obj = cur_obj
                neighbours = self.get_neighbours(best_solution, operator=self.operators_list[0])
            else:
                neighbours.pop(ni)
                if len(neighbours) == 0: # when the neighbour space empty, change anothor neighbour structure(operator)
                    operator_k += 1
                    if operator_k < len(self.operators_list):
                        operator = self.operators_list[operator_k]
                        neighbours = self.get_neighbours(best_solution, operator=operator)
                    else:
                        print('local optimal, break out, iterated {} times'.format(step))
                        break

            self.process.append(best_obj)
        self.best_solution = best_solution
        self.best_obj = best_obj
        self.best_routes = self.transfer(self.best_solution)
        return self.best_routes, self.best_obj


if __name__ == "__main__":
    file_name = "solomon_100\C101.txt"
    graph = GraphTool.Graph(file_name)
    iter_num = 100000
    alg = VNS(graph, iter_num)
    routes, obj = alg.run()
    obj = evaluate(graph, routes)
    print('obj: {}, {} vehicles in total'.format(obj, len(routes)))
    alg.draw(routes)
    alg.show_process()
