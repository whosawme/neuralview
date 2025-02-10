import pygame
import numpy as np
from typing import List, Tuple
import math

# Initialize Pygame
pygame.init()
pygame.font.init()

class Button:
    def __init__(self, x: int, y: int, width: int, height: int, text: str, is_toggle: bool = False):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = (100, 100, 100)
        self.hover_color = (150, 150, 150)
        self.text_color = (255, 255, 255)
        self.font = pygame.font.SysFont('Arial', 16)
        self.active = False
        self.is_toggle = is_toggle
        self.clicked = False  # For non-toggle buttons

    def draw(self, screen):
        color = self.hover_color if self.active else self.color
        pygame.draw.rect(screen, color, self.rect)
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if self.is_toggle:
                    self.active = not self.active
                else:
                    self.clicked = True
                return True
        if event.type == pygame.MOUSEBUTTONUP:
            self.clicked = False
        return False

class InputBox:
    def __init__(self, x: int, y: int, width: int, height: int, text: str = ''):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.active = False
        self.color_inactive = (100, 100, 100)
        self.color_active = (200, 200, 200)
        self.font = pygame.font.SysFont('Arial', 16)
        self.txt_surface = self.font.render(text, True, (255, 255, 255))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    return self.text
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                self.txt_surface = self.font.render(self.text, True, (255, 255, 255))
        return None

    def draw(self, screen):
        color = self.color_active if self.active else self.color_inactive
        pygame.draw.rect(screen, color, self.rect, 2)
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y + 5))

class Neuron:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.value = 0.0
        self.weights = []
        self.bias = np.random.randn()
        self.delta = 0.0
        
    def draw(self, screen, radius=15):
        # Map activation to color (brighter = more activated)
        intensity = int(255 * (1 / (1 + np.exp(-self.value))))
        color = (intensity, intensity, intensity)
        pygame.draw.circle(screen, color, (self.x, self.y), radius)
        pygame.draw.circle(screen, (200, 200, 200), (self.x, self.y), radius, 1)

class Layer:
    def __init__(self, num_neurons: int, y_start: int, x_pos: int):
        self.neurons = []
        spacing = 50
        total_height = (num_neurons - 1) * spacing
        start_y = y_start - total_height // 2
        
        for i in range(num_neurons):
            self.neurons.append(Neuron(x_pos, start_y + i * spacing))

    def draw(self, screen):
        for neuron in self.neurons:
            neuron.draw(screen)

class NeuralNetwork:
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.1):
        self.layers = []
        self.learning_rate = learning_rate
        
        # Calculate x positions for layers
        window_width = 800
        margin = 100
        layer_spacing = (window_width - 2 * margin) // (len(layer_sizes) - 1)
        
        # Create layers
        for i, size in enumerate(layer_sizes):
            x_pos = margin + i * layer_spacing
            self.layers.append(Layer(size, 300, x_pos))
        
        # Initialize weights
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            for neuron in current_layer.neurons:
                neuron.weights = np.random.randn(len(next_layer.neurons)) * 0.1

    def forward(self, inputs: np.ndarray):
        # Set input values
        for neuron, input_val in zip(self.layers[0].neurons, inputs):
            neuron.value = input_val
        
        # Forward propagation
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            for j, next_neuron in enumerate(next_layer.neurons):
                next_neuron.value = next_neuron.bias
                for current_neuron in current_layer.neurons:
                    next_neuron.value += current_neuron.value * current_neuron.weights[j]
                next_neuron.value = 1 / (1 + np.exp(-next_neuron.value))  # sigmoid

    def backward(self, targets: np.ndarray):
        # Output layer error
        output_layer = self.layers[-1]
        for i, neuron in enumerate(output_layer.neurons):
            error = targets[i] - neuron.value
            neuron.delta = error * neuron.value * (1 - neuron.value)
        
        # Hidden layers error
        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            for j, neuron in enumerate(layer.neurons):
                error = 0.0
                for k, next_neuron in enumerate(next_layer.neurons):
                    error += next_neuron.delta * neuron.weights[k]
                neuron.delta = error * neuron.value * (1 - neuron.value)
        
        # Update weights
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            for neuron in layer.neurons:
                for j, next_neuron in enumerate(next_layer.neurons):
                    neuron.weights[j] += self.learning_rate * next_neuron.delta * neuron.value
                    next_neuron.bias += self.learning_rate * next_neuron.delta

    def draw(self, screen):
        # Draw connections
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            for current_neuron in current_layer.neurons:
                for j, next_neuron in enumerate(next_layer.neurons):
                    weight = current_neuron.weights[j]
                    # Color based on weight (red negative, green positive)
                    if weight < 0:
                        color = (min(255, -weight * 255), 0, 0)
                    else:
                        color = (0, min(255, weight * 255), 0)
                    pygame.draw.line(screen, color, 
                                   (current_neuron.x, current_neuron.y),
                                   (next_neuron.x, next_neuron.y), 1)
        
        # Draw layers
        for layer in self.layers:
            layer.draw(screen)

class Visualizer:
    def __init__(self):
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Neural Network Visualizer")
        
        # Network configuration
        self.layer_sizes = [2, 4, 3, 2]  # Default architecture
        self.network = NeuralNetwork(self.layer_sizes)
        self.selected_layer = 0  # Currently selected layer for modification
        
        # UI Elements
        self.input_box = InputBox(10, 10, 200, 30, "0.5,0.5")
        self.target_box = InputBox(10, 50, 200, 30, "1,0")
        
        # Layer control buttons
        self.add_layer_btn = Button(10, 90, 30, 30, "+")
        self.remove_layer_btn = Button(50, 90, 30, 30, "-")
        self.prev_layer_btn = Button(90, 90, 30, 30, "←")
        self.next_layer_btn = Button(130, 90, 30, 30, "→")
        self.add_neuron_btn = Button(170, 90, 30, 30, "▲")
        self.remove_neuron_btn = Button(210, 90, 30, 30, "▼")
        
        # Training controls
        self.step_btn = Button(250, 90, 60, 30, "Step")
        
        self.font = pygame.font.SysFont('Arial', 16)
        self.clock = pygame.time.Clock()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            # Handle input boxes
            input_result = self.input_box.handle_event(event)
            target_result = self.target_box.handle_event(event)
            
            # Handle network structure buttons
            if self.add_layer_btn.handle_event(event):
                self.layer_sizes.insert(self.selected_layer + 1, 4)  # Add new layer with 4 neurons
                self.network = NeuralNetwork(self.layer_sizes)
                
            if self.remove_layer_btn.handle_event(event):
                if len(self.layer_sizes) > 2:  # Keep at least input and output layers
                    self.layer_sizes.pop(self.selected_layer)
                    self.selected_layer = min(self.selected_layer, len(self.layer_sizes) - 1)
                    self.network = NeuralNetwork(self.layer_sizes)
            
            if self.prev_layer_btn.handle_event(event):
                self.selected_layer = max(0, self.selected_layer - 1)
                
            if self.next_layer_btn.handle_event(event):
                self.selected_layer = min(len(self.layer_sizes) - 1, self.selected_layer + 1)
                
            if self.add_neuron_btn.handle_event(event):
                self.layer_sizes[self.selected_layer] += 1
                self.network = NeuralNetwork(self.layer_sizes)
                
            if self.remove_neuron_btn.handle_event(event):
                if self.layer_sizes[self.selected_layer] > 1:  # Keep at least one neuron per layer
                    self.layer_sizes[self.selected_layer] -= 1
                    self.network = NeuralNetwork(self.layer_sizes)
            
            # Handle keyboard controls
            if event.type == pygame.KEYDOWN:
                if not (self.input_box.active or self.target_box.active):  # Only if not typing in input boxes
                    if event.key == pygame.K_LEFT:
                        self.selected_layer = max(0, self.selected_layer - 1)
                    elif event.key == pygame.K_RIGHT:
                        self.selected_layer = min(len(self.layer_sizes) - 1, self.selected_layer + 1)
                    elif event.key == pygame.K_UP:
                        self.layer_sizes[self.selected_layer] += 1
                        self.network = NeuralNetwork(self.layer_sizes)
                    elif event.key == pygame.K_DOWN:
                        if self.layer_sizes[self.selected_layer] > 1:
                            self.layer_sizes[self.selected_layer] -= 1
                            self.network = NeuralNetwork(self.layer_sizes)
                    elif event.key == pygame.K_RETURN:
                        try:
                            inputs = np.array([float(x) for x in self.input_box.text.split(',')])
                            targets = np.array([float(x) for x in self.target_box.text.split(',')])
                            self.network.forward(inputs)
                            self.network.backward(targets)
                        except (ValueError, IndexError) as e:
                            print(f"Error processing input: {e}")
            
            # Handle training button
            if self.step_btn.handle_event(event):
                try:
                    inputs = np.array([float(x) for x in self.input_box.text.split(',')])
                    targets = np.array([float(x) for x in self.target_box.text.split(',')])
                    self.network.forward(inputs)
                    self.network.backward(targets)
                except (ValueError, IndexError) as e:
                    print(f"Error processing input: {e}")
        
        return True

    def draw(self):
        self.screen.fill((30, 30, 30))
        
        # Draw network
        self.network.draw(self.screen)
        
        # Draw UI elements
        self.input_box.draw(self.screen)
        self.target_box.draw(self.screen)
        self.step_btn.draw(self.screen)
        
        # Draw layer control buttons
        self.add_layer_btn.draw(self.screen)
        self.remove_layer_btn.draw(self.screen)
        self.prev_layer_btn.draw(self.screen)
        self.next_layer_btn.draw(self.screen)
        self.add_neuron_btn.draw(self.screen)
        self.remove_neuron_btn.draw(self.screen)
        
        # Draw labels
        input_label = self.font.render("Input values (comma-separated):", True, (255, 255, 255))
        target_label = self.font.render("Target values (comma-separated):", True, (255, 255, 255))
        layer_info = self.font.render(f"Layer {self.selected_layer + 1}: {self.layer_sizes[self.selected_layer]} neurons", True, (255, 255, 255))
        
        self.screen.blit(input_label, (10, -5))
        self.screen.blit(target_label, (10, 35))
        self.screen.blit(layer_info, (10, 75))
        
        # Highlight selected layer
        if 0 <= self.selected_layer < len(self.network.layers):
            layer = self.network.layers[self.selected_layer]
            for neuron in layer.neurons:
                pygame.draw.circle(self.screen, (255, 255, 0), (neuron.x, neuron.y), 18, 2)
        
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.draw()
            self.clock.tick(60)

if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.run()