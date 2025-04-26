class PostOffice:
    def __init__(self, agents: list):
        self.messages_manager = []
        self.agents = agents

    def add_message(self, message: dict):
        """
        message: dict
        {
            'sender': sender_id,
            'recipient': recipient_id,
            'content': content,
            'iteration': iteration
        }
        """
        self.messages_manager.append(message)

    def deliver_messages(self):
        for message in self.messages_manager:
            sender_id = message['sender']
            recipient_id = message['recipient']
            content = message['content']
            iteration = message['iteration']

            # print(f"Delivering message from Agent {sender_id} to Agent {recipient_id}: {content}")
            recipient_agent = next((agent for agent in self.agents if agent.id == recipient_id), None)
            recipient_agent.receive_message(sender_id, content, iteration)
        self.messages_manager.clear()