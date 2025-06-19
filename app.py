import streamlit as st
import asyncio
import os
import pickle
from typing import Dict, List, Optional
import tempfile

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.element
import ifcopenshell.util.shape
import numpy as np
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


# --- 1. App Configuration and Setup ---

st.set_page_config(page_title="IFC AI Assistant", layout="wide")
st.title("🤖 IFC AI Assistant")
st.caption("Your intelligent assistant for querying Industry Foundation Classes (IFC) files.")

# Load the list of all possible entity names from the pickle file once at startup.
@st.cache_data
def load_ifc_entities():
    try:
        with open(os.path.join(os.getcwd(), "ifc_entities.pkl"), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Fatal Error: 'ifc_entities.pkl' not found. Please add the file to the root directory.")
        return []

all_ifc_entities = load_ifc_entities()

# --- 2. Efficient, Cached IFC Model Loader ---

@st.cache_resource(show_spinner="Loading and caching IFC model...")
def get_ifc_model(path: str) -> ifcopenshell.file:
    """Loads an IFC model from a path using Streamlit's resource cache."""
    print(f"\n>> RESOURCE CACHE MISS: Loading IFC model from disk: {path}...")
    return ifcopenshell.open(path)


# Helper functions (remain unchanged)
def unpack_entity(entity):
    if not isinstance(entity, ifcopenshell.entity_instance):
        if isinstance(entity, (list, tuple)):
            return [unpack_entity(item) for item in entity]
        return entity
    return {key: unpack_entity(value) for key, value in entity.get_info().items()}

def create_shape(element):
    settings = ifcopenshell.geom.settings()
    return ifcopenshell.geom.create_shape(settings=settings, inst=element)


# --- 3. Agent Tools (Unchanged Core Logic) ---
# All tool functions are exactly the same as before.

@tool
async def list_all_materials(path: str, object_type: str) -> list:
    """List all materials related to a specific object type (e.g., IfcWall, IfcBeam)."""
    def _sync_logic():
        model = get_ifc_model(path)
        material_list = []
        for entity_name in all_ifc_entities:
            if object_type.lower() in entity_name.lower():
                for element in model.by_type(entity_name):
                    if not hasattr(element, "HasAssociations"): continue
                    for i in element.HasAssociations:
                        if i.is_a('IfcRelAssociatesMaterial'):
                            mat_rel = i.RelatingMaterial
                            if mat_rel.is_a('IfcMaterial'):
                                material_list.append((element.id(), element.Name, mat_rel.Name))
                            elif mat_rel.is_a('IfcMaterialList'):
                                for mat in mat_rel.Materials:
                                    material_list.append((element.id(), element.Name, mat.Name))
                            elif mat_rel.is_a('IfcMaterialLayerSetUsage'):
                                for mat_layer in mat_rel.ForLayerSet.MaterialLayers:
                                    material_list.append((element.id(), element.Name, mat_layer.Material.Name))
        return material_list
    return await asyncio.to_thread(_sync_logic)

@tool
async def list_all_types(path: str) -> list:
    """List all objects types in the IFC file"""
    def _sync_logic():
        model = get_ifc_model(path)
        type = model.by_type("IfcObject")
        return [e.get_info()['type'] for e in type]
    return await asyncio.to_thread(_sync_logic)

@tool
async def list_all_related_materials(path: str, object_id: int) -> list:
    """List all materials related to a specific object ID."""
    def _sync_logic():
        model = get_ifc_model(path)
        element = model.by_id(object_id)
        if element.HasAssociations:
                for i in element.HasAssociations:
                    if i.is_a('IfcRelAssociatesMaterial'):
                        if i.RelatingMaterial.is_a('IfcMaterial'):
                            return (element.id(),element.Name,i.RelatingMaterial.Name)
                        elif i.RelatingMaterial.is_a('IfcMaterialList'):
                            for materials in i.RelatingMaterial.Materials:
                                return [(element.id(),element.Name, materials.Name)]
                        elif i.RelatingMaterial.is_a('IfcMaterialLayerSetUsage'):
                            for materials in i.RelatingMaterial.ForLayerSet.MaterialLayers:
                                return[(element.id(),element.Name,materials.Material.Name)]
    return await asyncio.to_thread(_sync_logic)

@tool
async def get_objects_count(path: str, object_type: Optional[str] = None) -> Dict[str, int]:
    """Get the count of IFC objects that match a given object_type in the model."""
    def _sync_logic():
        if not object_type: return {}
        model = get_ifc_model(path)
        matched_counts = {}
        search_term = object_type.lower()
        for entity_name in all_ifc_entities:
            if search_term in entity_name.lower():
                count = len(model.by_type(entity_name))
                if count > 0:
                    matched_counts[entity_name] = count
        return matched_counts
    return await asyncio.to_thread(_sync_logic)


@tool
async def list_all_objects_names(path: str, object_type: str) -> list[str]:
    """List all unique 'Name' attribute values for a given object type."""
    def _sync_logic():
        model = get_ifc_model(path)
        matched_types = set()
        for entity_name in all_ifc_entities:
            if object_type.lower() in entity_name.lower():
                for element in model.by_type(entity_name):
                    if element.Name:
                        matched_types.add(element.Name)
        return sorted(list(matched_types))
    return await asyncio.to_thread(_sync_logic)

@tool
async def get_object_info(path: str, object_id: int) -> dict:
    """Get the detailed information for a single object by its ID."""
    def _sync_logic():
        model = get_ifc_model(path)
        entity = model.by_id(object_id)
        return unpack_entity(entity)
    return await asyncio.to_thread(_sync_logic)

@tool
async def get_object_dims(path: str, object_id: int) -> dict:
    """Get the 3 dimensions of a single object by its ID."""
    def _sync_logic():
        model = get_ifc_model(path)
        entity = model.by_id(object_id)
        if not entity: return {}
        shape = create_shape(entity)
        verts = ifcopenshell.util.shape.get_vertices(shape.geometry, is_2d=False)
        bbox = ifcopenshell.util.shape.get_bbox(verts)
        dims = np.round(np.abs(np.subtract(bbox[1], bbox[0])),2).tolist()
        return {"X-DIM": dims[0], "Y-DIM": dims[1], "Z-DIM": dims[2]}
    return await asyncio.to_thread(_sync_logic)

@tool
async def list_entities(path: str, object_type: str) -> list[int]:
    """List IDs of all entities of a given IFC class (e.g., IfcWall, IfcDoor)."""
    def _sync_logic():
        model = get_ifc_model(path)
        all_ids = []
        for entity_name in all_ifc_entities:
            if object_type.lower() in entity_name.lower():
                all_ids.extend([e.id() for e in model.by_type(entity_name)])
        return all_ids
    return await asyncio.to_thread(_sync_logic)

@tool
async def get_takeoffs(path: str, object_type: str) -> dict:
    """Return a dictionary of total quantities (area, volume) for an IFC entity type."""
    def _sync_logic():
        model = get_ifc_model(path)
        total_volume, total_area = 0.0, 0.0
        settings = ifcopenshell.geom.settings()
        for entity_name in all_ifc_entities:
            if object_type.lower() in entity_name.lower():
                for element in model.by_type(entity_name):
                    try:
                        shape = ifcopenshell.geom.create_shape(settings=settings, inst=element)
                        mesh = shape.geometry
                        total_volume += ifcopenshell.util.shape.get_volume(mesh)
                        total_area += ifcopenshell.util.shape.get_area(mesh)
                    except Exception:
                        continue
        return {"total_area": total_area, "total_volume": total_volume}
    return await asyncio.to_thread(_sync_logic)

@tool
async def get_object_geometry_properties(path: str, object_id: int) -> dict:
    """Get an object's quantity takeoffs from its property sets (QTOs)."""
    def _sync_logic():
        model = get_ifc_model(path)
        entity = model.by_id(object_id)
        return ifcopenshell.util.element.get_psets(entity, qtos_only=True)
    return await asyncio.to_thread(_sync_logic)

@tool
async def save_to_local_file(file_name: str = "output.txt", data: str = "") -> str:
    """Save text to a local file. Returns the file path."""
    def _sync_logic():
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "w") as f:
            f.write(data)
        return f"File saved to {file_path}"
    return await asyncio.to_thread(_sync_logic)

@tool
async def calculate_distance_between_2_shapes(path:str, entity_id_1:int, entity_id_2:int):
    """Calculate the distance between two shapes"""
    def _sync_logic():
        model = get_ifc_model(path)
        element1, element2 = model.by_id(entity_id_1), model.by_id(entity_id_2)
        ctx = ifcopenshell.geom.settings()
        shape1 = ifcopenshell.geom.create_shape(ctx, element1)
        shape2 = ifcopenshell.geom.create_shape(ctx, element2)
        vert_1 = ifcopenshell.util.shape.get_shape_bbox_centroid(shape1,shape1.geometry).tolist()
        vert_2 = ifcopenshell.util.shape.get_shape_bbox_centroid(shape2,shape2.geometry).tolist()
        abs_subtract = np.subtract(vert_1, vert_2)
        subtract_sq = np.square(abs_subtract)
        return np.sqrt(sum(subtract_sq))
    return await asyncio.to_thread(_sync_logic)

@tool
async def find_nearby_elements(path:str,  entity_id:int, range:float):
    """Find nearby elements within a given range"""
    def _sync_logic():
        model = get_ifc_model(path)
        element = model.by_id(entity_id)
        shape = create_shape(element)
        source_box_centroid = ifcopenshell.util.shape.get_shape_bbox_centroid(shape,shape.geometry)
        nearby_elements = []
        for e in model.by_type("IfcObject"):
            if isinstance(e, ifcopenshell.entity_instance):
                try:
                    target_shape = create_shape(e)
                    if target_shape is None: continue
                    target_bbox_centroid = ifcopenshell.util.shape.get_shape_bbox_centroid(target_shape, target_shape.geometry)
                    calculated_dist = np.sqrt(np.sum(np.square(np.subtract(source_box_centroid, target_bbox_centroid))))
                    if calculated_dist <= range and calculated_dist > 0:
                        nearby_elements.append((e.id(), e.Name, e.is_a(), calculated_dist ))
                except RuntimeError:
                    continue
        return nearby_elements
    return await asyncio.to_thread(_sync_logic)

@tool
async def get_min_max_3dcoords(path:str, entity_id:int):
    """Get the minimum and maximum 3D coordinates of a shape"""
    def _sync_logic():
        model = get_ifc_model(path)
        element = model.by_id(entity_id)
        context = ifcopenshell.geom.settings()
        shape = ifcopenshell.geom.create_shape(context, element)
        mesh = shape.geometry
        vert = ifcopenshell.util.shape.get_vertices(mesh)
        max_coord = np.amax(vert, 0)
        min_coord = np.amin(vert, 0)
        return {"max_coord":max_coord.tolist(), "min_coord":min_coord.tolist()}
    return await asyncio.to_thread(_sync_logic)

# --- 4. Agent and Executor Setup ---

@st.cache_resource(show_spinner="Initializing AI Agent...")
def get_agent_executor(google_api_key: str, file_path: str):
    """Creates and caches the LangChain agent and executor."""
    os.environ["GOOGLE_API_KEY"] = google_api_key

    tools = [
        list_entities, list_all_objects_names, list_all_types, list_all_materials,
        list_all_related_materials, get_object_info, get_object_dims,
        get_min_max_3dcoords, get_objects_count, get_takeoffs,
        get_object_geometry_properties, calculate_distance_between_2_shapes,
        find_nearby_elements, save_to_local_file
    ]

    prompt_template = f"""You are a helpful assistant for querying data from Industry Foundation Classes (IFC) files.
You must use the provided tools coherently with your intellectual faculties to answer questions. 
The user has uploaded an IFC file. When you use any tool that requires a 'path' argument, you MUST use the following file path: '{file_path}'
Do not ask the user for the path; use the one provided here.
Return all results in markdown format and use tables as you see fit.
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    chat_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-05-20",
        temperature=0.7,
    )

    agent = create_tool_calling_agent(chat_model, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


# --- 5. Streamlit UI and Chat Logic ---

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    google_api_key = st.text_input("Google API Key", type="password", help="Get your key from https://aistudio.google.com/app/apikey")
    uploaded_file = st.file_uploader("Upload an IFC file", type=["ifc"])

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Handle file upload
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ifc") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        st.session_state.ifc_path = tmp_file.name
    st.sidebar.success(f"Loaded `{uploaded_file.name}`")
    get_ifc_model(st.session_state.ifc_path)
else:
    st.session_state.ifc_path = None

# **** CHANGE 1: SIMPLIFIED MESSAGE DISPLAY LOOP ****
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat input logic
if prompt := st.chat_input("Ask a question about your IFC file..."):
    if not google_api_key:
        st.info("Please enter your Google API Key in the sidebar to continue.")
        st.stop()
    if not st.session_state.get("ifc_path"):
        st.info("Please upload an IFC file in the sidebar to begin.")
        st.stop()

    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the cached agent executor
    agent_executor = get_agent_executor(google_api_key, st.session_state.ifc_path)

    # Display assistant response and agent thoughts
    with st.chat_message("assistant"):
        # The StreamlitCallbackHandler will automatically display agent thoughts
        st_callback = StreamlitCallbackHandler(st.container())
        
        with st.spinner("AI is thinking..."):
            response = asyncio.run(agent_executor.ainvoke(
                {
                    "input": prompt,
                    "chat_history": st.session_state.chat_history,
                },
                {"callbacks": [st_callback]}
            ))

        output = response['output']
        st.markdown(output) # Display the final answer

        # **** CHANGE 2: SIMPLIFIED MESSAGE SAVING ****
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": output})
        
        # Add messages to the LangChain chat history object
        st.session_state.chat_history.extend([
            HumanMessage(content=prompt),
            AIMessage(content=output)
        ])
