class LinkedListNode:
    def __init__(self, data=None):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_LinkedListNode = LinkedListNode(data)
        if self.head is None:
            self.head = new_LinkedListNode
            return
        last_LinkedListNode = self.head
        while last_LinkedListNode.next is not None:
            last_LinkedListNode = last_LinkedListNode.next
        last_LinkedListNode.next = new_LinkedListNode

    def prepend(self, data):
        new_LinkedListNode = LinkedListNode(data)
        new_LinkedListNode.next = self.head
        self.head = new_LinkedListNode

    def print_list(self):
        current_LinkedListNode = self.head
        while current_LinkedListNode is not None:
            print(current_LinkedListNode.data)
            current_LinkedListNode = current_LinkedListNode.next
