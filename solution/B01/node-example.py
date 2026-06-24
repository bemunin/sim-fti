import rclpy
from rclpy.node import Node


class RealTimeParamNode(Node):
    def __init__(self):
        super().__init__('real_time_param_node')

        # 1. Declare the parameter
        self.declare_parameter('max_speed', 1.5)
        self.get_logger().info('Node started. Waiting for real-time parameter changes...')

        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        # 2. Read the parameter INSIDE the callback loop
        # This guarantees you are grabbing the freshest value every single second.
        current_speed = self.get_parameter('max_speed').get_parameter_value().double_value

        # 3. Use the dynamic value
        self.get_logger().info(f'Moving robot at dynamic speed: {current_speed}')


def main(args=None):
    rclpy.init(args=args)
    node = RealTimeParamNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()