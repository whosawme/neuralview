import numpy as np
from typing import List, Tuple
import math
import pygame
import pyperclip

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
    def __init__(self, layer_sizes=None, learning_rate=0.1):
        # Default network structure
        layer_sizes = layer_sizes or [2, 4, 3, 2]
        
        self.layers = []
        self.learning_rate = learning_rate
        self.components = []
        
        # Calculate layer positions
        window_width = 800
        margin = 100
        layer_spacing = (window_width - 2 * margin) // (len(layer_sizes) - 1)
        
        # Create layers with positioning
        for i, size in enumerate(layer_sizes):
            x_pos = margin + i * layer_spacing
            self.layers.append(Layer(size, 300, x_pos))
        
        # Initialize weights (He initialization)
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            for neuron in current_layer.neurons:
                std = np.sqrt(2.0 / len(current_layer.neurons))
                neuron.weights = np.random.randn(len(next_layer.neurons)) * std
                neuron.bias = 0.0

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, inputs):
        # Set input values
        for neuron, input_val in zip(self.layers[0].neurons, inputs):
            neuron.value = input_val
        
        # Forward propagation with ReLU activation
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            for j, next_neuron in enumerate(next_layer.neurons):
                next_neuron.value = 0
                for current_neuron in current_layer.neurons:
                    next_neuron.value += current_neuron.value * current_neuron.weights[j]
                next_neuron.value = self.relu(next_neuron.value)

    def backward(self, targets):
        # Calculate output layer error
        output_layer = self.layers[-1]
        output_deltas = []
        
        for i, neuron in enumerate(output_layer.neurons):
            error = 2 * (neuron.value - targets[i])
            delta = error * self.relu_derivative(neuron.value)
            neuron.delta = delta
            output_deltas.append(delta)
        
        # Backpropagate error
        for i in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            for j, current_neuron in enumerate(current_layer.neurons):
                error = 0.0
                for k, next_neuron in enumerate(next_layer.neurons):
                    error += next_neuron.delta * current_neuron.weights[k]
                current_neuron.delta = error * self.relu_derivative(current_neuron.value)
        
        # Update weights
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            for current_neuron in current_layer.neurons:
                for j, next_neuron in enumerate(next_layer.neurons):
                    weight_grad = current_neuron.value * next_neuron.delta
                    current_neuron.weights[j] -= self.learning_rate * weight_grad

    def get_pytorch_code(self):
        current_layer_sizes = [len(layer.neurons) for layer in self.layers]
        code = [
            "import torch",
            "import torch.nn as nn",
            "",
            "class NeuralNetwork(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            f"        # Network architecture: {current_layer_sizes}",
            ""
        ]
        
        # Add layers
        for i in range(len(current_layer_sizes) - 1):
            code.append(f"        self.layer{i+1} = nn.Linear({current_layer_sizes[i]}, {current_layer_sizes[i+1]})")
            if i < len(current_layer_sizes) - 2:
                code.append(f"        self.relu{i+1} = nn.ReLU()")
        
        code.extend([
            "",
            "    def forward(self, x):",
            "        x = self.layer1(x)"
        ])
        
        # Add forward pass
        for i in range(1, len(current_layer_sizes) - 1):
            code.append(f"        x = self.relu{i}(x)")
            code.append(f"        x = self.layer{i+1}(x)")
        
        code.extend([
            "        return x",
            "",
            "# Create model instance",
            "model = NeuralNetwork()",
            "",
            "# Example training loop",
            "criterion = nn.MSELoss()",
            "optimizer = torch.optim.SGD(model.parameters(), lr=0.1)",
        ])
        
        return "\n".join(code)

        def draw(self, screen):
            # Draw connections
            for i in range(len(self.layers) - 1):
                current_layer = self.layers[i]
                next_layer = self.layers[i + 1]
                
                for current_neuron in current_layer.neurons:
                    for j, next_neuron in enumerate(next_layer.neurons):
                        weight = current_neuron.weights[j]
                        color = (min(255, -weight * 255), 0, 0) if weight < 0 else (0, min(255, weight * 255), 0)
                        pygame.draw.line(screen, color, 
                                    (current_neuron.x, current_neuron.y),
                                    (next_neuron.x, next_neuron.y), 1)
            
            # Draw layers
            for layer in self.layers:
                layer.draw(screen)

    def draw(self, screen):
        # Draw connections
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            for current_neuron in current_layer.neurons:
                for j, next_neuron in enumerate(next_layer.neurons):
                    weight = current_neuron.weights[j]
                    color = (min(255, -weight * 255), 0, 0) if weight < 0 else (0, min(255, weight * 255), 0)
                    pygame.draw.line(screen, color, 
                                   (current_neuron.x, current_neuron.y),
                                   (next_neuron.x, next_neuron.y), 1)
        
        # Draw layers
        for layer in self.layers:
            layer.draw(screen)


class ScrollablePane:
    def __init__(self, rect, font):
        self.rect = rect
        self.font = font
        self.scroll_y = 0
        self.content = []
        self.max_scroll = 0
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:  # Mouse wheel up
                self.scroll_y = max(0, self.scroll_y - 20)
            elif event.button == 5:  # Mouse wheel down
                self.scroll_y = min(self.max_scroll, self.scroll_y + 20)

    def draw(self, screen, content):
        # Draw background
        pygame.draw.rect(screen, (40, 40, 40), self.rect)
        # pygame.draw.rect(screen, (100, 100, 100), self.rect, 2)
        
        # Calculate content height and max scroll
        total_height = len(content) * 20
        self.max_scroll = max(0, total_height - self.rect.height)
        
        # Draw visible content
        visible_start = self.scroll_y // 20
        visible_end = min(len(content), (self.scroll_y + self.rect.height) // 20 + 1)
        
        for i, line in enumerate(content[visible_start:visible_end]):
            y_pos = self.rect.y + i * 20 - (self.scroll_y % 20)
            if self.rect.y <= y_pos <= self.rect.bottom:
                text = self.font.render(line, True, (200, 200, 200))
                screen.blit(text, (self.rect.x + 10, y_pos))

class CodeWindow:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.is_maximized = False
        self.maximize_btn = Button(x + width - 30, y, 30, 30, "□")
        self.original_dims = (x, y, width, height)

    def draw(self, screen, code_lines):
        if self.is_maximized:
            self.rect = pygame.Rect(0, 80, screen.get_width(), screen.get_height() - 80)
        self.maximize_btn.draw(screen)
        # Draw code content...

    def handle_event(self, event):
        if self.maximize_btn.handle_event(event):
            self.is_maximized = not self.is_maximized
            if not self.is_maximized:
                self.rect = pygame.Rect(*self.original_dims)


class NetworkWindow:
    def __init__(self, network):
        self.screen = pygame.display.set_mode((800, 600))
        self.network = network
        self.running = True

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            self.screen.fill((30, 30, 30))
            self.network.draw(self.screen)
            pygame.display.flip()




class CopyButton:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.font = pygame.font.SysFont('Arial', 12)
        
    def draw(self, screen):
        pygame.draw.rect(screen, (60, 60, 60), self.rect)
        text = self.font.render("Copy", True, (200, 200, 200))
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)
        
    def handle_event(self, event, text_to_copy):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                import pyperclip
                pyperclip.copy(text_to_copy)
                return True
        return False


class Visualizer:
    def __init__(self):
        self.width = 1200
        self.height = 800
        
        # Define layer sizes early
        self.layer_sizes = [2, 4, 3, 2]  # Default architecture
        # Create basic network after layer_sizes is defined
        self.basic_network = NeuralNetwork(self.layer_sizes)
        
        self.network = NeuralNetwork(self.layer_sizes)
        self.component_types = ["Convolutional", "LSTM", "Transformer"]
        self.selected_component = None
        self.component_params = {}
        self.input_boxes = {}
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.create_component_selection_ui()
        self.create_parameter_input_ui()
        
        
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Neural Network Visualizer")

        self.code_font = pygame.font.SysFont('Courier', 14)
        self.code_lines = []
        self.update_code_preview()

        # Network configuration (for basic layers - keep separate for now)
        
        
        self.selected_layer = 0  # Currently selected layer for modification

        # UI Elements (for basic layers)
        # self.input_box = InputBox(10, 10, 200, 30, "0.5,0.5")
        # self.target_box = InputBox(10, 50, 200, 30, "1,0")
        self.data_menu = DataInputMenu(10, 50, 200, 200)  # After banner

        # Banner dimensions
        banner_height = 40
        self.banner = pygame.Rect(0, 0, self.width, banner_height)

        # Second banner (component controls)
        self.component_banner = pygame.Rect(0, banner_height, self.width, banner_height)

        # Add to Visualizer:
        self.network_popup_btn = Button(700, banner_height + 5, 30, 30, "↗")

        # Calculate button spacing to fit banner width
        total_buttons = 8  # Number of buttons
        button_width = 30  # Standard button width
        step_button_width = 60  # Width for step button
        total_width = button_width * (total_buttons - 1) + step_button_width
        spacing = (self.width - total_width) / (total_buttons + 1) /3

        # Position buttons
        x = spacing
        y = (banner_height - 30) / 2  # Center buttons vertically in banner
        self.add_layer_btn = Button(x, y, button_width, 30, "+")
        x += button_width + spacing
        self.remove_layer_btn = Button(x, y, button_width, 30, "-")
        x += button_width + spacing
        self.prev_layer_btn = Button(x, y, button_width, 30, "←")
        x += button_width + spacing
        self.next_layer_btn = Button(x, y, button_width, 30, "→")
        x += button_width + spacing
        self.add_neuron_btn = Button(x, y, button_width, 30, "▲")
        x += button_width + spacing
        self.remove_neuron_btn = Button(x, y, button_width, 30, "▼")
        x += button_width + spacing
        self.step_btn = Button(x, y, step_button_width, 30, "Step")
        x += step_button_width + spacing
        self.tensor_toggle_btn = Button(x, y, button_width, 30, "T")


        # Training controls (for basic layers)
        # self.step_btn = Button(250, 90, 60, 30, "Step")

        self.code_pane_coordinates = [800, 44, 380, 680]
        self.tensor_pane_coordinates = [00, 600, 380, 380]

        self.code_pane = ScrollablePane(pygame.Rect(*self.code_pane_coordinates), self.code_font)
        self.tensor_pane = ScrollablePane(pygame.Rect(*self.tensor_pane_coordinates), self.code_font)
        self.copy_button = CopyButton(1150, 15, 50, 20)
        self.show_tensors = False  # Toggle for tensor view

        self.font = pygame.font.SysFont('Arial', 16)

        self.component_buttons = []
        x_start, y_start = 10, 10

        # Move data menu and component buttons to second banner
        component_spacing = self.width / 6
        x = component_spacing / 2
        y = banner_height + (banner_height - 30) / 2
        self.data_menu = DataInputMenu(x, y, 150, 200)
        x += component_spacing
        for component_type in self.component_types:
            button = Button(x, y, 100, 30, component_type)
            self.component_buttons.append(button)
            x += component_spacing

        self.add_component_button = Button(x, y, 100, 30, "Add Component")
        self.selected_component_type = None


    def create_parameter_input_ui(self):
        self.component_params = {  # Define parameters for each component type
            "Convolutional": {"kernel_size": 3, "output_channels": 16},  # Example defaults
            "LSTM": {"input_size": 10, "hidden_size": 64},
            "Transformer": {"input_dim": 64, "head_dim": 32, "ff_dim": 128}
        }
        self.input_boxes = {}

        y_start = 120  # Start position for input boxes (below buttons)
        x_start = 10

        for component_type, params in self.component_params.items():
            self.input_boxes[component_type] = {}  # Create nested dictionary
            current_y = y_start  # Reset y_start for each component type
            for param_name, default_value in params.items():
                input_box = InputBox(x_start, current_y, 50, 30, str(default_value))
                self.input_boxes[component_type][param_name] = input_box
                current_y += 40  # Spacing between input boxes


    def update_code_preview(self):
        if hasattr(self, 'network'):  # Changed from basic_network to network
            self.code_lines = self.network.get_pytorch_code().split('\n')
        else:
            self.code_lines = []


    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            self.data_menu.handle_event(event)

            if self.network_popup_btn.handle_event(event):
                network_window = NetworkWindow(self.network)
                network_window.run()
                
            # Toggle tensor view with 'T' key or button
            if (event.type == pygame.KEYDOWN and event.key == pygame.K_t) or \
            (self.tensor_toggle_btn.handle_event(event)):
                self.show_tensors = not self.show_tensors

            # Handle pane scrolling
            self.code_pane.handle_event(event)
            self.tensor_pane.handle_event(event)
            
            # Handle copy button
            self.copy_button.handle_event(event, "\n".join(self.code_lines))

            # Component selection handling
            for button in self.component_buttons:
                if button.handle_event(event):
                    component = NetworkComponent(button.text, {}, 0, 0)
                    component.update_network(self.network)
                    self.update_code_preview()  # Refresh code display
            
            if self.add_component_button.handle_event(event) and self.selected_component_type:
                try:
                    params = {
                        k: float(v.text) 
                        for k, v in self.input_boxes[self.selected_component_type].items()
                    }
                    x_pos = 100 + len(self.network.components) * 150
                    y_pos = 300

                    component_map = {
                        "Convolutional": ConvolutionalComponent,
                        "LSTM": LSTMComponent,
                        "Transformer": TransformerComponent
                    }

                    new_component = component_map[self.selected_component_type](params, x_pos, y_pos)
                    self.network.components.append(new_component)
                except Exception as e:
                    print(f"Error adding component: {e}")

            if self.copy_button.handle_event(event, "\n".join(self.code_lines)):
                print("Code copied to clipboard!")
            
            # Toggle tensor view with 'T' key
            if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                self.show_tensors = not self.show_tensors
            

            
            # Handle network structure buttons
            if self.add_layer_btn.handle_event(event):
                self.layer_sizes.insert(self.selected_layer + 1, 4)
                self.network = NeuralNetwork(self.layer_sizes)
                self.update_code_preview()
                
            if self.remove_layer_btn.handle_event(event):
                if len(self.layer_sizes) > 2:
                    self.layer_sizes.pop(self.selected_layer)
                    self.selected_layer = min(self.selected_layer, len(self.layer_sizes) - 1)
                    self.network = NeuralNetwork(self.layer_sizes)
                    self.update_code_preview()

            if self.prev_layer_btn.handle_event(event):
                self.selected_layer = max(0, self.selected_layer - 1)
                
            if self.next_layer_btn.handle_event(event):
                self.selected_layer = min(len(self.layer_sizes) - 1, self.selected_layer + 1)
                
            if self.add_neuron_btn.handle_event(event):
                self.layer_sizes[self.selected_layer] += 1
                self.network = NeuralNetwork(self.layer_sizes)
                self.update_code_preview()
                
            if self.remove_neuron_btn.handle_event(event):
                if self.layer_sizes[self.selected_layer] > 1:
                    self.layer_sizes[self.selected_layer] -= 1
                    self.network = NeuralNetwork(self.layer_sizes)
                    self.update_code_preview()
            
            # Handle keyboard controls
            if event.type == pygame.KEYDOWN and not (self.data_menu.input_box.active or self.data_menu.target_box.active):
                if event.key == pygame.K_LEFT:
                    self.selected_layer = max(0, self.selected_layer - 1)
                elif event.key == pygame.K_RIGHT:
                    self.selected_layer = min(len(self.layer_sizes) - 1, self.selected_layer + 1)
                elif event.key == pygame.K_UP:
                    self.layer_sizes[self.selected_layer] += 1
                    self.network = NeuralNetwork(self.layer_sizes)
                    self.update_code_preview()
                elif event.key == pygame.K_DOWN:
                    if self.layer_sizes[self.selected_layer] > 1:
                        self.layer_sizes[self.selected_layer] -= 1
                        self.network = NeuralNetwork(self.layer_sizes)
                        self.update_code_preview()
                elif event.key == pygame.K_RETURN:
                    try:
                        inputs = np.array([float(x) for x in self.data_menu.input_box.text.split(',')])
                        targets = np.array([float(x) for x in self.data_menu.target_box.text.split(',')])
                        if len(inputs) != self.layer_sizes[0] or len(targets) != self.layer_sizes[-1]:
                            print(f"Error: Input size must be {self.layer_sizes[0]} and target size must be {self.layer_sizes[-1]}")
                            return
                        self.network.forward(inputs)
                        self.network.backward(targets)
                    except (ValueError, IndexError) as e:
                        print(f"Error processing input: {e}")
            
            # Handle training button
            if self.step_btn.handle_event(event):
                try:
                    inputs = np.array([float(x) for x in self.data_menu.input_box.text.split(',')])
                    targets = np.array([float(x) for x in self.data_menu.target_box.text.split(',')])
                    if len(inputs) != self.layer_sizes[0] or len(targets) != self.layer_sizes[-1]:
                        print(f"Error: Input size must be {self.layer_sizes[0]} and target size must be {self.layer_sizes[-1]}")
                        return
                    self.network.forward(inputs)
                    self.network.backward(targets)
                except (ValueError, IndexError) as e:
                    print(f"Error processing input: {e}")


        
        
        return True


    def create_component_selection_ui(self):
        self.component_buttons = []
        x_start, y_start = 10, 10
        for component_type in self.component_types:
            button = Button(x_start, y_start, 100, 30, component_type)
            self.component_buttons.append(button)
            x_start += 110

        # Add "Add Component" button
        self.add_component_button = Button(10, 80, 150, 30, "Add Component")
        self.selected_component_type = None


    def draw(self):

        self.screen.fill((30, 30, 30))
        # Draw banner
        pygame.draw.rect(self.screen, (50, 50, 50), self.banner)
        
        #draw du input menu
        self.data_menu.draw(self.screen)

        # Draw network
        self.network.draw(self.screen)
        
        # Draw UI elements
        self.step_btn.draw(self.screen)
        
        # Draw layer control buttons
        self.add_layer_btn.draw(self.screen)
        self.remove_layer_btn.draw(self.screen)
        self.prev_layer_btn.draw(self.screen)
        self.next_layer_btn.draw(self.screen)
        self.add_neuron_btn.draw(self.screen)
        self.remove_neuron_btn.draw(self.screen)
        self.network_popup_btn.draw(self.screen)
        
        # Draw labels
        # input_label = self.font.render("Input values (comma-separated):", True, (255, 255, 255))
        # target_label = self.font.render("Target values (comma-separated):", True, (255, 255, 255))
        layer_info = self.font.render(f"Layer {self.selected_layer + 1}: {self.layer_sizes[self.selected_layer]} neurons", True, (255, 255, 255))
        
        # self.screen.blit(input_label, (10, -5))
        # self.screen.blit(target_label, (10, 35))
        self.screen.blit(layer_info, (10, 75))

        # Draw code pane with scroll
        title = self.font.render("PyTorch Pane:", True, (255, 255, 255))
        self.screen.blit(title, (810, 15))
        self.code_pane.draw(self.screen, self.code_lines)
        self.copy_button.draw(self.screen)

        
        # Highlight selected layer
        if 0 <= self.selected_layer < len(self.network.layers):
            layer = self.network.layers[self.selected_layer]
            for neuron in layer.neurons:
                pygame.draw.circle(self.screen, (255, 255, 0), (neuron.x, neuron.y), 18, 2)
        
                
        # Draw tensor visualization if enabled
        if self.show_tensors:
            tensor_title = self.font.render("Tensor Operations:", True, (255, 255, 255))
            self.screen.blit(tensor_title, (10, 605))
            tensor_content = self.get_tensor_visualization()
            self.tensor_pane.draw(self.screen, tensor_content)

        for button in self.component_buttons:
            button.draw(self.screen)
            self.add_component_button.draw(self.screen)

        self.tensor_toggle_btn.draw(self.screen)

        pygame.display.flip()

    def get_tensor_visualization(self):
        lines = []
        for i, layer in enumerate(self.network.layers[:-1]):
            next_layer = self.network.layers[i + 1]
            lines.append(f"Layer {i+1} -> {i+2}:")
            lines.append(f"Input shape: ({len(layer.neurons)}, 1)")
            lines.append(f"Weight shape: ({len(layer.neurons)}, {len(next_layer.neurons)})")
            lines.append(f"Output shape: ({len(next_layer.neurons)}, 1)")
            
            # Show actual values
            lines.append("\nInput values:")
            input_vals = [f"{n.value:.3f}" for n in layer.neurons]
            lines.append(f"[{', '.join(input_vals)}]")
            
            lines.append("\nWeight matrix:")
            for n in layer.neurons:
                weight_str = [f"{w:.3f}" for w in n.weights]
                lines.append(f"[{', '.join(weight_str)}]")
                
            lines.append("\nActivations:")
            activations = [f"{n.value:.3f}" for n in next_layer.neurons]
            lines.append(f"[{', '.join(activations)}]")
            lines.append("-" * 40 + "\n")
        return lines
        


    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.draw()
            self.clock.tick(60)


class DataInputMenu:
    def __init__(self, x, y, width, height):
        self.closed_rect = pygame.Rect(x, y, width, 30)
        self.open_rect = pygame.Rect(x, y, width, height)
        self.is_open = False
        self.input_box = InputBox(x + 10, y + 40, width - 20, 30, "0.5,0.5")
        self.target_box = InputBox(x + 10, y + 100, width - 20, 30, "1,0")
        self.upload_btn = Button(x + 10, y + 160, width - 20, 30, "Upload File")
        
    def draw(self, screen):
        if self.is_open:
            pygame.draw.rect(screen, (50, 50, 50), self.open_rect)
            self.input_box.draw(screen)
            self.target_box.draw(screen)
            self.upload_btn.draw(screen)
            # Labels
            font = pygame.font.SysFont('Arial', 12)
            input_label = font.render("Input values:", True, (255, 255, 255))
            target_label = font.render("Target values:", True, (255, 255, 255))
            screen.blit(input_label, (self.open_rect.x + 10, self.open_rect.y + 25))
            screen.blit(target_label, (self.open_rect.x + 10, self.open_rect.y + 85))
        
        # Always draw header
        pygame.draw.rect(screen, (70, 70, 70), self.closed_rect)
        font = pygame.font.SysFont('Arial', 14)
        text = font.render("Data Input ▼" if not self.is_open else "Data Input ▲", True, (255, 255, 255))
        screen.blit(text, (self.closed_rect.x + 10, self.closed_rect.y + 8))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.closed_rect.collidepoint(event.pos):
                self.is_open = not self.is_open
                return True
        
        if self.is_open:
            input_result = self.input_box.handle_event(event)
            target_result = self.target_box.handle_event(event)
            if self.upload_btn.handle_event(event):
                # Handle file upload
                pass
            return input_result or target_result
        return False



class ConvolutionalLayer:  # If you *really* need this, refactor it
    def __init__(self, in_channels, out_channels, y_start, x_pos):  # Simplified
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.neurons = []
        self.weights = np.random.randn(in_channels, out_channels) * np.sqrt(2.0 / in_channels)  # Weights matrix
        self.biases = np.random.randn(out_channels)

        spacing = 50
        total_height = (out_channels - 1) * spacing
        start_y = y_start - total_height // 2
        for i in range(out_channels):
            self.neurons.append(Neuron(x_pos, start_y + i * spacing))

    def forward(self, x):  # x shape: (in_channels,)
        output = np.dot(x, self.weights) + self.biases
        return output

    def draw(self, screen):
        for neuron in self.neurons:
            neuron.draw(screen) #Return attention weights for visualization
    
    # def forward(self, input_data):  # input_data shape: (height, width, in_channels)
    #     height, width, _ = input_data.shape
    #     output_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
    #     output_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1
    #     output = np.zeros((output_height, output_width, self.out_channels))

    #     for c_out in range(self.out_channels):
    #         for y in range(output_height):
    #             for x in range(output_width):
    #                 y_start = y * self.stride - self.padding
    #                 y_end = y_start + self.kernel_size
    #                 x_start = x * self.stride - self.padding
    #                 x_end = x_start + self.kernel_size

    #                 input_patch = input_data[max(0, y_start):min(height, y_end), max(0, x_start):min(width, x_end)]
                    
    #                 #Handle padding
    #                 padded_patch = np.zeros((self.kernel_size, self.kernel_size, self.in_channels))
    #                 y_patch_start = max(0, -y_start)
    #                 x_patch_start = max(0, -x_start)
    #                 padded_patch[y_patch_start:y_patch_start+input_patch.shape[0], x_patch_start:x_patch_start+input_patch.shape[1]] = input_patch
                    
    #                 output[y, x, c_out] = np.sum(padded_patch * self.kernels[c_out]) + self.biases[c_out]
    #     return output
    
class LSTMCell:
    def __init__(self, input_size, hidden_size, y_start, x_pos):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.neurons = []
        # Initialize weights and biases (example: Xavier initialization)
        self.W_f = np.random.randn(hidden_size + input_size, hidden_size) * np.sqrt(2.0 / (hidden_size + input_size))  # Forget gate weights
        self.U_f = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b_f = np.random.randn(hidden_size)

        self.W_i = np.random.randn(hidden_size + input_size, hidden_size) * np.sqrt(2.0 / (hidden_size + input_size))  # Input gate weights
        self.U_i = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b_i = np.random.randn(hidden_size)

        self.W_c = np.random.randn(hidden_size + input_size, hidden_size) * np.sqrt(2.0 / (hidden_size + input_size))  # Cell gate weights
        self.U_c = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b_c = np.random.randn(hidden_size)

        self.W_o = np.random.randn(hidden_size + input_size, hidden_size) * np.sqrt(2.0 / (hidden_size + input_size))  # Output gate weights
        self.U_o = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b_o = np.random.randn(hidden_size)

        #Create neurons for hidden state
        spacing = 50
        total_height = (hidden_size - 1) * spacing
        start_y = y_start - total_height // 2
        for i in range(hidden_size):
            self.neurons.append(Neuron(x_pos, start_y + i * spacing))  # Reuse Neuron class

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, h_prev, c_prev):
        combined = np.concatenate((x, h_prev))

        f = self.sigmoid(np.dot(combined, self.W_f) + np.dot(h_prev, self.U_f) + self.b_f)  # Forget gate
        i = self.sigmoid(np.dot(combined, self.W_i) + np.dot(h_prev, self.U_i) + self.b_i)  # Input gate
        c_tilde = self.tanh(np.dot(combined, self.W_c) + np.dot(h_prev, self.U_c) + self.b_c)  # Candidate cell state
        c = f * c_prev + i * c_tilde  # New cell state
        o = self.sigmoid(np.dot(combined, self.W_o) + np.dot(h_prev, self.U_o) + self.b_o)  # Output gate
        h = o * self.tanh(c)  # New hidden state

        return h, c

    def draw(self, screen):
        for neuron in self.neurons:
            neuron.draw(screen)
            #Visualize attention weights (simplified)
            #TODO: Make this work with sequences
            # for i in range(len(attention_weights)):
            #     for j in range(len(attention_weights)):
            #         x1 = self.neurons[i].x
            #         y1 = self.neurons[i].y
            #         x2 = self.neurons[j].x + 50 #Offset for visualisation
            #         y2 = self.neurons[j].y
            #         weight = attention_weights[i,j]
            #         color = (0, min(255, weight * 255), 0)
            #         pygame.draw.line(screen, color, (x1, y1), (x2, y2), 1)
 

class NetworkComponent:  # Base class
    def __init__(self, component_type, params, x_pos, y_pos):
        self.component_type = component_type
        self.params = params
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.output_shape = None #Placeholder for output shape calculation

    def draw(self, screen):
        # Placeholder - draw component-specific UI
        font = pygame.font.SysFont('Arial', 16)
        text = font.render(self.component_type, True, (255, 255, 255))
        screen.blit(text, (self.x_pos, self.y_pos))  # Basic placement

    def forward(self, x):
        raise NotImplementedError("Forward pass not implemented for this component")

    def backward(self, error):
        raise NotImplementedError("Backward pass not implemented for this component")
    

    def update_network(self, network):
        if self.component_type == "Transformer":
            network.layers.append(TransformerBlock(...))
        elif self.component_type == "CNN":
            network.layers.append(ConvolutionalLayer(...))
        elif self.component_type == "LSTM":
            network.layers.append(LSTMCell(...))

class ConvolutionalComponent(NetworkComponent):
    def __init__(self, params, x_pos, y_pos):
        super().__init__("Convolutional", params, x_pos, y_pos)
        self.kernel = np.random.rand(params["kernel_size"], params["kernel_size"]) #Example kernel
        self.output_shape = (params["output_channels"],) #Example output shape

    def draw(self, screen):
        super().draw(screen)
        pygame.draw.rect(screen, (0, 0, 255), (self.x_pos + 20, self.y_pos + 10, 50, 30))

    def forward(self, x):
        #Implement your convolutional forward pass here
        #Example (replace with your actual convolution logic)
        return np.random.rand(self.output_shape[0])

    def backward(self, error):
        #Implement your convolutional backward pass here
        pass #Placeholder

class LSTMComponent(NetworkComponent):
    def __init__(self, params, x_pos, y_pos):
        super().__init__("LSTM", params, x_pos, y_pos)
        self.output_shape = (params["hidden_size"],) #Example output shape

    def draw(self, screen):
        super().draw(screen)
        pygame.draw.rect(screen, (0, 255, 0), (self.x_pos + 20, self.y_pos + 10, 50, 30))

    def forward(self, x):
        #Implement your LSTM forward pass here
        return np.random.rand(self.output_shape[0])

    def backward(self, error):
        #Implement your LSTM backward pass here
        pass #Placeholder

class TransformerComponent(NetworkComponent):
    def __init__(self, params, x_pos, y_pos):
        super().__init__("Transformer", params, x_pos, y_pos)
        self.output_shape = (params["input_dim"],) #Example output shape

    def draw(self, screen):
        super().draw(screen)
        pygame.draw.rect(screen, (255, 0, 0), (self.x_pos + 20, self.y_pos + 10, 50, 30))

    def forward(self, x):
        #Implement your transformer forward pass here
        return np.random.rand(self.output_shape[0])

    def backward(self, error):
        #Implement your transformer backward pass here
        pass #Placeholder




def softmax(x, axis=-1):
    """Compute softmax probabilities."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # For numerical stability
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class TransformerBlock:
    def __init__(self, input_dim, head_dim, ff_dim, y_start, x_pos, dropout=0.1):  # Correct order
        self.attention_head = AttentionHead(input_dim, head_dim, y_start, x_pos)
        self.feed_forward = FeedForward(input_dim, ff_dim, y_start + 100, x_pos)  # Offset for visualization
        self.norm1 = LayerNormalization(input_dim)
        self.norm2 = LayerNormalization(input_dim)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x):
        attn_output, attention_weights = self.attention_head.forward(x)
        x = self.norm1(x + self.dropout1(attn_output))  # Residual connection and dropout
        ff_output = self.feed_forward.forward(x)
        x = self.norm2(x + self.dropout2(ff_output))  # Residual connection and dropout
        return x, attention_weights  # Return attention weights for visualization

    def draw(self, screen):
        self.attention_head.draw(screen)
        self.feed_forward.draw(screen)


class FeedForward:
    def __init__(self, input_dim, hidden_dim, y_start, x_pos):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.neurons = []
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.random.randn(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.random.randn(input_dim)

        #Neurons for visualizing the output
        spacing = 50
        total_height = (input_dim - 1) * spacing #Output is same dimension as input
        start_y = y_start - total_height // 2
        for i in range(input_dim):
            self.neurons.append(Neuron(x_pos, start_y + i * spacing))  # Reuse Neuron class


    def forward(self, x):
        x = np.dot(x, self.W1) + self.b1
        x = relu(x)
        x = np.dot(x, self.W2) + self.b2
        return x

    def draw(self, screen):
        for neuron in self.neurons:
            neuron.draw(screen)

class LayerNormalization:
    def __init__(self, features, eps=1e-6):
        self.gamma = np.ones(features)
        self.beta = np.zeros(features)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / (std + self.eps)
        return self.gamma * x_norm + self.beta


class Dropout:
    def __init__(self, rate):
        self.rate = rate

    def forward(self, x):
        if self.rate > 0.:
            mask = np.random.binomial(1, 1 - self.rate, size=x.shape)
            return x * mask / (1 - self.rate)  # Inverted dropout
        return x
    

def relu(x):
    return np.maximum(0, x)




if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.run()