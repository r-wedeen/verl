# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Kimina tool for LEAN4 proof verification.
"""

import json
from transformers.utils import get_json_schema
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema, ToolResponse
from kimina_client import AsyncKiminaClient


class KiminaTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        # Initialize Kimina client
        server_url = config.get("kimina_server_url", "http://localhost:8000")
        api_key = config.get("kimina_api_key", None)
        self.client = AsyncKiminaClient(server_url, api_key=api_key)
    
    async def verify_lean_code(self, proof: str, custom_id: str = None, infotree_type: str = "original") -> str:
        """Verify Lean 4 code/proof using Kimina.
        
        Args:
            proof: The Lean 4 code or proof to verify (e.g., "#check Nat")
            custom_id: Optional custom identifier for this verification
            infotree_type: Type of infotree to use (default: "original")
            
        Returns:
            str: The verification result from Kimina
        """
        # Use the Kimina client's check method
        result = await self.client.check(proof)

        # Format the response more cleanly
        if isinstance(result, list) and len(result) > 0:
            repl_response = result[0]
            
            # Convert ReplResponse object to dict if needed
            if not isinstance(repl_response, dict):
                # Try Pydantic model_dump() first
                if hasattr(repl_response, "model_dump"):
                    repl_response = repl_response.model_dump()
                # Otherwise try __dict__ or vars()
                elif hasattr(repl_response, "__dict__"):
                    repl_response = repl_response.__dict__
                else:
                    repl_response = {}
            
            response = repl_response.get("response", {})
            
            # Convert response to dict if it's an object
            if not isinstance(response, dict):
                if hasattr(response, "model_dump"):
                    response = response.model_dump()
                elif hasattr(response, "__dict__"):
                    response = response.__dict__
                else:
                    response = {}
            
            messages = response.get("messages", [])
            
            formatted_messages = []
            for msg in messages:
                # Convert message to dict if it's an object
                if not isinstance(msg, dict):
                    if hasattr(msg, "model_dump"):
                        msg = msg.model_dump()
                    elif hasattr(msg, "__dict__"):
                        msg = msg.__dict__
                    else:
                        continue
                
                severity = msg.get("severity", "info")
                data = msg.get("data", "")
                if data:  # Only add if there's actual data
                    formatted_messages.append(f"[{severity.upper()}] {data}")
            
            if formatted_messages:
                return "\n".join(formatted_messages)
        
        # Fallback: return JSON or string representation
        return json.dumps(result) if isinstance(result, dict) else str(result)
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        schema = get_json_schema(self.verify_lean_code)
        return OpenAIFunctionToolSchema(**schema)
    
    async def execute(self, instance_id: str, parameters: dict, **kwargs) -> tuple[ToolResponse, float, dict]:
        try:
            proof = parameters.get("proof", "")
            custom_id = parameters.get("custom_id", None)
            infotree_type = parameters.get("infotree_type", "original")
            
            result = await self.verify_lean_code(proof, custom_id=custom_id, infotree_type=infotree_type)
            return ToolResponse(text=result), 0.0, {}
        except Exception as e:
            return ToolResponse(text=f"Error: {str(e)}"), 0.0, {}

