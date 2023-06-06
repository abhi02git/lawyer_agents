import os
import streamlit as st
import databutton as db
from apikeys import openai_api_key, serpapi_api_key
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from PyPDF2 import PdfReader


class LawyerAgent:
    def __init__(self, system_message: SystemMessage, model: ChatOpenAI) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def step(self, input_message: HumanMessage) -> AIMessage:
        messages = self.update_messages(input_message)
        output_message = self.model(messages)
        self.update_messages(output_message)
        return output_message


class CAMELApp:
    def __init__(self):
        self.supporting_lawyer = "Supporting lawyer agent"
        self.opposing_lawyer = "Opposing lawyer agent"
        self.word_limit = 50
        self.text = ""
        self.specified_task = ""
        self.supporting_sys_msg = None
        self.opposing_sys_msg = None

    def run(self):
        self.initialize_case_file()
        self.initialize_task_specifier()
        self.initialize_agents()
        self.initialize_chats()

    def initialize_case_file(self):
        # case_content = st.text_input("Enter the details of the case")
        pdf = st.file_uploader("Upload your pdf", type= "pdf")
        
        if pdf:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                self.text += page.extract_text()

    def initialize_task_specifier(self):
        case = self.text

        task_specifier_sys_msg = SystemMessage(content="Make arguments in the case which must be true within law")
        task_specifier_prompt = (
            """Here is a case that {supporting_lawyer} will argue against {opposing_lawyer} CASE: {case}.
            Make arguments which are true within the law in bullet points.
            {supporting_lawyer} will make arguments to support the case and make arguments opposite to {opposing_lawyer} to prove the client innocent.
            {opposing_lawyer} will make arguments against the {supporting_lawyer} to prove the client guilty.
            Please make it more specific. Be creative and imaginative.
            Please reply with the specified task in {word_limit} words or less. Do not add anything else."""
        )
        task_specifier_template = HumanMessagePromptTemplate.from_template(template=task_specifier_prompt)
        task_specify_agent = LawyerAgent(task_specifier_sys_msg, ChatOpenAI(temperature=0.9))
        task_specifier_msg = task_specifier_template.format_messages(
            supporting_lawyer=self.supporting_lawyer,
            opposing_lawyer=self.opposing_lawyer,
            case=case,
            word_limit=self.word_limit
        )[0]
        specified_task_msg = task_specify_agent.step(task_specifier_msg)
        self.specified_task = specified_task_msg.content
        print(f"Specified task: {self.specified_task}")

    def initialize_agents(self):
        case = self.text

        supporting_agent_inception_prompt = (
            """Never forget you are a {supporting_lawyer} and I am a {opposing_lawyer}. Never flip roles! Never support me!
            We must always make opposite arguments.
            Always make arguments in bullet points.
            You are a very experienced senior lawyer who makes arguments which are within law.
            Here is the case: {case}. Never forget our case!

            You must write arguments which support the client and provide specific arguments which oppose the {opposing_lawyer} arguments.
            You must decline honestly if you cannot make opposing arguments due to physical or legal reasons or your capability and explain the reasons.
            Make arguments to prove the client innocent and not guilty.
            Do not add anything else other than your arguments.
            You are always supposed to ask me any questions.
            """
        )

        opposing_agent_inception_prompt = (
            """Never forget you are a {opposing_lawyer} and I am a {supporting_lawyer}. Never flip roles! You will always oppose me.
            We must always make opposite arguments.
            Always make arguments in bullet points.
            You are a very experienced senior lawyer who makes arguments which are within law.
            Here is the case: {case}. Never forget our case!

            You must write arguments which oppose the client and provide specific arguments which oppose the {supporting_lawyer} arguments.
            You must decline honestly if you cannot make opposing arguments due to physical or legal reasons or your capability and explain the reasons.
            Make arguments to prove the client guilty.
            Do not add anything else other than your arguments.
            You are always supposed to ask me any questions."""
        )

        supporting_sys_template = SystemMessagePromptTemplate.from_template(template=supporting_agent_inception_prompt)
        supporting_sys_msg = supporting_sys_template.format_messages(
            supporting_lawyer=self.supporting_lawyer,
            opposing_lawyer=self.opposing_lawyer,
            case=case
        )[0]

        opposing_sys_template = SystemMessagePromptTemplate.from_template(template=opposing_agent_inception_prompt)
        opposing_sys_msg = opposing_sys_template.format_messages(
            supporting_lawyer=self.supporting_lawyer,
            opposing_lawyer=self.opposing_lawyer,
            case=case
        )[0]

        self.supporting_lawyer_agent = LawyerAgent(supporting_sys_msg, ChatOpenAI(temperature=0.2))
        self.opposing_lawyer_agent = LawyerAgent(opposing_sys_msg, ChatOpenAI(temperature=0.2))

        # Reset agents
        self.supporting_lawyer_agent.reset()
        self.opposing_lawyer_agent.reset()

    def initialize_chats(self):
        chat_turn_limit, n = 10, 0
        supporting_msg = HumanMessage(content=f"{self.opposing_sys_msg}")
        while n < chat_turn_limit:
            n += 1
            user_ai_msg = self.opposing_lawyer_agent.step(supporting_msg)
            opposing_msg = HumanMessage(content=user_ai_msg.content)
            st.write(f"({self.opposing_lawyer}):\n\n{opposing_msg.content}\n\n")

            assistant_ai_msg = self.supporting_lawyer_agent.step(opposing_msg)
            supporting_msg = HumanMessage(content=assistant_ai_msg.content)
            st.write(f"({self.supporting_lawyer}):\n\n{supporting_msg.content}\n\n")
            # if "<CAMEL_TASK_DONE>" in opposing_msg.content:
                # break


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = openai_api_key
    camel_app = CAMELApp()
    camel_app.run()