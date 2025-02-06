import os
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents import create_csv_agent
from langchain.callbacks.tracers import ConsoleCallbackHandler


load_dotenv()

def main():
    print("start...")

    instructions="""you are an agent designed to write and execute python code to answer question.
    you have access to python REPL , which you can use to execute python code.
    
    To create a directory run os.makedirs() code.

    If you get an error, debug your code and try again.
    Only use output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write the code to answer the question, just return "I don't know" as an answer.
    """

    base_prompt= hub.pull('langchain-ai/react-agent-template')
    prompt= base_prompt.partial(instructions= instructions)

    tools=[PythonREPLTool()]
    agent= create_react_agent(
        prompt=prompt,
        llm= ChatOpenAI(temperature=0, model= "gpt-4-turbo"),
        tools=tools
    )

    python_agent_executor= AgentExecutor(agent=agent, tools=tools, verbose=True)

    # python_agent_executor.invoke(
    #     input= {
    #         "input": """generate and save in current directory 15 QRcodes that point to 
    #         www.udemy.com/course/langchain, you have qrcode package installed already"""
    #     }
    # )


    csv_agent=create_csv_agent(
        llm= ChatOpenAI(temperature=0, model='gpt-4'),
        path= "episode_info.csv",
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True
    )

    # csv_agent.invoke(
    #     input={
    #         "input": "how many columns are there in file episode_info.csv"
    #     }
    # )

    # csv_agent.invoke(
    #     input={
    #         "input": "who write the most episodes? how many episodes did he write?"
    #     },
    #     config={"callbacks":[ConsoleCallbackHandler()]} # used bcoz -> ValueError: An output parsing error occurred. In order to pass this error back to the agent and have it try again,
    # )

    # csv_agent.invoke(
    #     input={
    #         "input": "which season has the most episodes? how many episodes are there?"
    #     },
    #     config={"callbacks":[ConsoleCallbackHandler()]}
    # )

    csv_agent.invoke(
            input={
                "input": "print seasons by number of episodes they have in ascending order."
            },
            config={"callbacks":[ConsoleCallbackHandler()]}
        )

if __name__== "__main__":
    main()