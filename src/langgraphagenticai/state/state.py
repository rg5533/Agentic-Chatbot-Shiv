from typing_extensions import TypedDict,List, Annotated
from langgraph.graph.message import add_messages


class State(TypedDict):
    """
     Represent the structure of the state used in graph
    """

    messages: Annotated[List,add_messages]

