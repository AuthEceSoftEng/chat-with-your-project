# chat-with-your-project
Tool-grounded conversational agent for software repositories

To deploy the agent you need access to the MongoDB database gddata by filling the MONGO_URI variable in .env.

Then create a virtual environment and install the required python packages.
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```
In tool_registry.py comment in or out the tools the configuration can have access to.

Start the MCP Server:
```
python mcp_server.py
```
Start an interface of the chatbot:
``` 
python mcp_main_cli.py
```
or in streamlit for GUI:
```
streamlit run mcp_main_gui.py
```

To replicate the dataset use instead test.py with the following configurations:

change system_prompts/system_prompt to the one desired
change output_dir paths to those desired

For brute force configuration use test_brute_force.py

To use KG a local Memgraph instance must be deployed.
To transform a MongoDB repository to a Memgraph one, use mongo_to_kg.py in KG folder.
