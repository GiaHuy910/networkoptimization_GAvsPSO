# -*- coding: utf-8 -*-
import random
import copy
import numpy as np
import pandas as pd
from collections import deque

class CompleteMaxFlowGA:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.population = []
        self.fitness_list = []
        self.best_flow = None
        self.best_flow_value = 0
        self.generation = 0
        self.generation_history = []      # Lịch sử thế hệ
        self.best_fitness_history = []    # Lịch sử best fitness mỗi thế hệ
        self.node_mapping = {}
        self.capacity_matrix = None
        self.all_paths = []

    def read_network_from_dataframe(self, df):
        """
        Đọc mạng lưới từ DataFrame và tạo ma trận capacity
        """
        all_nodes = set()
        
        from_col = df.columns[0]
        to_col = df.columns[1]
        capacity_col = df.columns[2] if len(df.columns) > 2 else None
        
        for _, row in df.iterrows():
            all_nodes.add(str(row[from_col]))
            all_nodes.add(str(row[to_col]))
        
        nodes_list = sorted(list(all_nodes))
        n = len(nodes_list)
        
        # Tạo ma trận capacity
        capacity_matrix = np.zeros((n, n))
        node_to_index = {node: idx for idx, node in enumerate(nodes_list)}
        index_to_node = {idx: node for node, idx in node_to_index.items()}
        
        self.node_mapping = {
            'node_to_index': node_to_index,
            'index_to_node': index_to_node,
            'nodes': nodes_list
        }
        
        # Điền giá trị capacity vào ma trận
        for _, row in df.iterrows():
            from_node = str(row[from_col])
            to_node = str(row[to_col])
            capacity = float(row[capacity_col]) if capacity_col else 1.0
            
            from_idx = node_to_index[from_node]
            to_idx = node_to_index[to_node]
            capacity_matrix[from_idx][to_idx] = capacity
        
        self.capacity_matrix = capacity_matrix
        return capacity_matrix

    def _convert_node_to_index(self, node):
        """Chuyển đổi node thành chỉ số"""
        if isinstance(node, int):
            return node
        elif node.isdigit():
            return int(node)
        else:
            if node in self.node_mapping['node_to_index']:
                return self.node_mapping['node_to_index'][node]
            else:
                raise ValueError(f"Node '{node}' not found in graph")

    def _convert_path_to_names(self, path):
        """Chuyển đổi đường đi từ chỉ số sang tên node"""
        if not path:
            return []
        return [self.node_mapping['index_to_node'].get(node, str(node)) for node in path]

    def _find_all_possible_paths(self, source, sink, max_paths=50):
        """Tìm tất cả các đường đi có thể từ source đến sink"""
        source_idx = self._convert_node_to_index(source)
        sink_idx = self._convert_node_to_index(sink)
        
        paths = []
        n = self.capacity_matrix.shape[0]
        
        def dfs(current, path, visited):
            if current == sink_idx:
                paths.append(path.copy())
                return
            
            for next_node in range(n):
                if (self.capacity_matrix[current][next_node] > 0 and 
                    not visited[next_node] and 
                    len(paths) < max_paths):
                    visited[next_node] = True
                    path.append(next_node)
                    dfs(next_node, path, visited)
                    path.pop()
                    visited[next_node] = False
        
        visited = [False] * n
        visited[source_idx] = True
        dfs(source_idx, [source_idx], visited)
        
        return paths

    def initialize_population(self, source, sink, population_size):
        """Khởi tạo quần thể - mỗi cá thể là một tập các đường đi với luồng"""
        population = []
        n = self.capacity_matrix.shape[0]
        source_idx = self._convert_node_to_index(source)
        sink_idx = self._convert_node_to_index(sink)
        
        # Tìm tất cả các đường đi có thể
        if not self.all_paths:
            self.all_paths = self._find_all_possible_paths(source, sink)
            if self.verbose:
                print(f"Found {len(self.all_paths)} possible paths from {source} to {sink}")
        
        for _ in range(population_size):
            # Mỗi cá thể là một dictionary: {path_index: flow}
            individual = {}
            residual_capacity = self.capacity_matrix.copy()
            total_flow = 0
            
            # Chọn ngẫu nhiên các đường đi và phân phối luồng
            num_paths = random.randint(3, min(10, len(self.all_paths)))
            selected_paths = random.sample(range(len(self.all_paths)), num_paths)
            
            for path_idx in selected_paths:
                path = self.all_paths[path_idx]
                
                # Tìm capacity nhỏ nhất trên đường đi
                path_capacity = float('inf')
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    path_capacity = min(path_capacity, residual_capacity[u][v])
                
                if path_capacity > 1e-6:
                    # Phân phối luồng ngẫu nhiên
                    flow = random.uniform(0.1, 0.8) * path_capacity
                    individual[path_idx] = flow
                    total_flow += flow
                    
                    # Cập nhật residual capacity
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        residual_capacity[u][v] -= flow
            
            population.append(individual)
        
        return population

    def calculate_fitness(self, population, source, sink):
        """Tính fitness - giá trị luồng từ source đến sink với ràng buộc bảo toàn luồng"""
        fitness_list = []
        source_idx = self._convert_node_to_index(source)
        sink_idx = self._convert_node_to_index(sink)
        
        for individual in population:
            if self._is_valid_flow_distribution(individual, source_idx, sink_idx):
                # Tính tổng luồng (luồng ra từ source)
                total_flow = self._calculate_total_flow(individual, source_idx)
                fitness_list.append(total_flow)
            else:
                fitness_list.append(0.0)
        
        return fitness_list

    def _is_valid_flow_distribution(self, individual, source, sink):
        """Kiểm tra tính hợp lệ của phân phối luồng"""
        n = self.capacity_matrix.shape[0]
        
        # Tính luồng trên từng cạnh
        edge_flows = np.zeros((n, n))
        
        for path_idx, flow in individual.items():
            path = self.all_paths[path_idx]
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_flows[u][v] += flow
        
        # Kiểm tra capacity constraints
        for i in range(n):
            for j in range(n):
                if edge_flows[i][j] > self.capacity_matrix[i][j] + 1e-6:
                    return False
        
        # Kiểm tra flow conservation (bảo toàn luồng)
        for node in range(n):
            if node != source and node != sink:
                flow_in = np.sum(edge_flows[:, node])   # Luồng vào
                flow_out = np.sum(edge_flows[node, :])  # Luồng ra
                if abs(flow_in - flow_out) > 1e-6:
                    return False
        
        return True

    def _calculate_total_flow(self, individual, source):
        """Tính tổng luồng từ source"""
        total_flow = 0.0
        
        for path_idx, flow in individual.items():
            path = self.all_paths[path_idx]
            if path[0] == source:  # Đường đi bắt đầu từ source
                total_flow += flow
        
        return total_flow

    def _calculate_flow_matrix(self, individual):
        """Tính ma trận luồng từ individual"""
        n = self.capacity_matrix.shape[0]
        flow_matrix = np.zeros((n, n))
        
        for path_idx, flow in individual.items():
            path = self.all_paths[path_idx]
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                flow_matrix[u][v] += flow
        
        return flow_matrix

    def selection(self, population, fitness_list, method='tournament'):
        """Chọn lọc cá thể"""
        if method == 'tournament':
            return self._tournament_selection(population, fitness_list)
        elif method == 'roulette':
            return self._roulette_selection(population, fitness_list)
        else:
            return self._tournament_selection(population, fitness_list)

    def _tournament_selection(self, population, fitness_list, tournament_size=3):
        """Chọn lọc tournament"""
        new_population = []
        for _ in range(len(population)):
            indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_list[i] for i in indices]
            best_index = indices[np.argmax(tournament_fitness)]
            new_population.append(copy.deepcopy(population[best_index]))
        return new_population

    def _roulette_selection(self, population, fitness_list):
        """Chọn lọc roulette wheel"""
        fitness_list = [max(0.001, f) for f in fitness_list]
        total_fitness = sum(fitness_list)

        if total_fitness == 0:
            return [copy.deepcopy(ind) for ind in population]

        probabilities = [fitness / total_fitness for fitness in fitness_list]
        selected_indices = np.random.choice(
            range(len(population)),
            size=len(population),
            p=probabilities
        )
        return [copy.deepcopy(population[i]) for i in selected_indices]

    def crossover(self, population, crossover_rate=0.8):
        """Lai ghép các cá thể"""
        new_population = []
        random.shuffle(population)

        for i in range(0, len(population), 2):
            if i + 1 < len(population) and random.random() < crossover_rate:
                parent1 = population[i]
                parent2 = population[i + 1]
                
                child1, child2 = self._crossover_individuals(parent1, parent2)
                new_population.extend([child1, child2])
            else:
                if i < len(population):
                    new_population.append(copy.deepcopy(population[i]))
                if i + 1 < len(population):
                    new_population.append(copy.deepcopy(population[i + 1]))

        return new_population

    def _crossover_individuals(self, ind1, ind2):
        """Lai ghép hai cá thể"""
        child1 = {}
        child2 = {}
        
        # Tất cả các đường đi có thể
        all_paths = set(ind1.keys()) | set(ind2.keys())
        
        for path_idx in all_paths:
            flow1 = ind1.get(path_idx, 0)
            flow2 = ind2.get(path_idx, 0)
            
            # Lai ghép tuyến tính
            alpha = random.random()
            child1_flow = alpha * flow1 + (1 - alpha) * flow2
            child2_flow = (1 - alpha) * flow1 + alpha * flow2
            
            # Chỉ giữ lại các đường đi có luồng đáng kể
            if child1_flow > 1e-6:
                child1[path_idx] = child1_flow
            if child2_flow > 1e-6:
                child2[path_idx] = child2_flow
        
        return child1, child2

    def mutation(self, population, mutation_rate=0.1):
        """Đột biến các cá thể"""
        new_population = []
        
        for individual in population:
            if random.random() < mutation_rate:
                mutated_ind = self._mutate_individual(individual)
                new_population.append(mutated_ind)
            else:
                new_population.append(copy.deepcopy(individual))
        return new_population

    def _mutate_individual(self, individual):
        """Đột biến một cá thể"""
        mutated = copy.deepcopy(individual)
        
        # Loại bỏ ngẫu nhiên một số đường đi
        if mutated and random.random() < 0.3:
            path_to_remove = random.choice(list(mutated.keys()))
            del mutated[path_to_remove]
        
        # Thêm đường đi mới
        if random.random() < 0.4 and self.all_paths:
            available_paths = set(range(len(self.all_paths))) - set(mutated.keys())
            if available_paths:
                new_path = random.choice(list(available_paths))
                mutated[new_path] = random.uniform(0.1, 1.0)
        
        # Thay đổi luồng trên các đường đi hiện có
        for path_idx in list(mutated.keys()):
            if random.random() < 0.5:
                # Thay đổi luồng
                change = random.uniform(-0.5, 0.5) * mutated[path_idx]
                new_flow = max(0, mutated[path_idx] + change)
                if new_flow > 1e-6:
                    mutated[path_idx] = new_flow
                else:
                    del mutated[path_idx]
        
        return mutated

    def genetic_algorithm(self, source, sink, population_size=50, max_generations=100, 
                        mutation_rate=0.1, crossover_rate=0.8):
        """Giải thuật Di Truyền chính cho Max Flow"""
        source_idx = self._convert_node_to_index(source)
        sink_idx = self._convert_node_to_index(sink)
        
        self._reset_history()
        
        # Khởi tạo quần thể
        population = self.initialize_population(source, sink, population_size)
        
        if self.verbose:
            print(f"\n=== MAX FLOW GENETIC ALGORITHM ===")
            print(f"Population size: {len(population)}")
            print(f"Graph size: {self.capacity_matrix.shape[0]} nodes")
            print(f"Source: {source} (index {source_idx})")
            print(f"Sink: {sink} (index {sink_idx})")
            print(f"Total capacity: {np.sum(self.capacity_matrix):.2f}")
            print(f"Number of possible paths: {len(self.all_paths)}")

        best_individual = None
        best_flow_value = 0

        for generation in range(max_generations):
            # Tính fitness
            fitness_list = self.calculate_fitness(population, source, sink)
            
            if not fitness_list or max(fitness_list) == 0:
                if self.verbose:
                    print(f"Generation {generation}: No valid flows found. Reinitializing population.")
                population = self.initialize_population(source, sink, population_size)
                fitness_list = self.calculate_fitness(population, source, sink)
                if not fitness_list or max(fitness_list) == 0:
                    if self.verbose:
                        print("Cannot initialize valid population. Stopping.")
                    break

            # LƯU LỊCH SỬ - CHỈ best fitness
            best_fitness = max(fitness_list)
            self._save_generation_history(generation, best_fitness)

            best_index = np.argmax(fitness_list)
            current_flow_value = fitness_list[best_index]

            if current_flow_value > best_flow_value:
                best_flow_value = current_flow_value
                best_individual = copy.deepcopy(population[best_index])
                if self.verbose:
                    print(f"Generation {generation}: New best flow value = {best_flow_value:.4f}")

            # Tiến hóa
            population = self.selection(population, fitness_list, 'tournament')
            population = self.crossover(population, crossover_rate)
            population = self.mutation(population, mutation_rate)

            # Giữ lại cá thể tốt nhất
            if best_individual is not None and len(population) > 0:
                population[0] = copy.deepcopy(best_individual)

        return best_individual, best_flow_value

    def _save_generation_history(self, generation, best_fitness):
        """Lưu lịch sử thế hệ và best fitness"""
        self.generation_history.append(generation)
        self.best_fitness_history.append(best_fitness)

    def _reset_history(self):
        """Reset lịch sử"""
        self.generation_history = []
        self.best_fitness_history = []

    def extract_all_paths_with_flows(self, individual):
        """Trích xuất tất cả các đường đi với luồng từ individual"""
        paths_with_flows = []
        
        for path_idx, flow in individual.items():
            if flow > 1e-6:  # Chỉ lấy các đường đi có luồng đáng kể
                path = self.all_paths[path_idx]
                path_names = self._convert_path_to_names(path)
                paths_with_flows.append({
                    'path': path,
                    'path_names': path_names,
                    'flow': flow,
                    'path_string': ' -> '.join(path_names)
                })
        
        # Sắp xếp theo luồng giảm dần
        paths_with_flows.sort(key=lambda x: x['flow'], reverse=True)
        return paths_with_flows

    def print_fitness_history(self):
        """In lịch sử best fitness của TẤT CẢ thế hệ"""
        if not self.best_fitness_history:
            print("No fitness history available!")
            return
        
        print(f"\nBEST FITNESS HISTORY (ALL GENERATIONS):")
        print("=" * 40)
        print("Generation | Best Fitness")
        print("-" * 40)
        
        # IN TẤT CẢ thế hệ
        for i in range(len(self.generation_history)):
            gen = self.generation_history[i]
            fitness = self.best_fitness_history[i]
            print(f"{gen:9d}  | {fitness:12.4f}")

    def run_complete_max_flow_ga(self, df, source, sink, population_size=50, max_generations=100,
                            mutation_rate=0.1, crossover_rate=0.8,verbose=True):
        """Chạy thuật toán Max Flow GA hoàn chỉnh và trả về tất cả đường đi"""
        
        self.verbose = verbose
        if self.verbose:
            print("=" * 60)
            print("COMPLETE MAX FLOW GENETIC ALGORITHM")
            print("=" * 60)

        # Đọc mạng lưới
        capacity_matrix = self.read_network_from_dataframe(df)
        
        if self.verbose:
            print(f"Network capacity matrix shape: {capacity_matrix.shape}")
            print(f"Available nodes: {self.node_mapping['nodes']}")
            print(f"Source: {source}, Sink: {sink}")
        
        # Chạy thuật toán
        best_individual, best_flow_value = self.genetic_algorithm(
            source=source,
            sink=sink,
            population_size=population_size,
            max_generations=max_generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate
        )

        # Trích xuất tất cả các đường đi
        all_paths_with_flows = []
        if best_individual is not None:
            all_paths_with_flows = self.extract_all_paths_with_flows(best_individual)

        # Hiển thị kết quả
        if self.verbose:
            print(f"\nFINAL RESULTS:")
            print(f"Max Flow Value: {best_flow_value:.4f}")
            print(f"Total paths found: {len(all_paths_with_flows)}")
            
            # In lịch sử fitness
            self.print_fitness_history()
            
            if all_paths_with_flows:
                print(f"\nALL PATHS WITH FLOWS:")
                print("=" * 80)
                for i, path_info in enumerate(all_paths_with_flows, 1):
                    print(f"Path {i}:")
                    print(f"  Flow: {path_info['flow']:.4f}")
                    print(f"  Route: {path_info['path_string']}")
                    print()

        return all_paths_with_flows, best_flow_value, self.generation_history, self.best_fitness_history