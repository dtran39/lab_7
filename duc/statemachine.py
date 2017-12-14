class StateMachine:

    def __init__(self):
        self.states = {}
        self.startState = None
        self.endStates=[]

    def addState(self, item):
        self.states[item.getName()] = item

    def setStartState(self, item):
        self.startState = item.getName()

    def run(self, info):
        try:
            handler = self.startState
            print("handler", handler)
        except:
            print("Start State Not defined")
        while True:



            item = self.states.get(handler)
            #print(item)
            (newState, info) = item.run(info)
            if newState in self.endStates:
                print("End")
            else:
                #print(handler," ", newState)
                if handler is not newState:
                    info.say_text("BEEP").wait_for_completed()
                    print("switching to ", newState, " from ", handler)
                handler = self.states[newState].getName()
                #print("Handler new ", handler)





