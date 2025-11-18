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
Reward function for LEAN4 proof verification using Kimina.
Returns 1.0 if proof is correct, 0.0 otherwise.
"""

import asyncio
import re
from typing import Optional
from kimina_client import AsyncKiminaClient


async def _verify_lean_proof_async(proof: str, kimina_server_url: str = "http://localhost:8000", api_key: Optional[str] = None) -> bool:
    """
    Verify a LEAN4 proof using Kimina client.
    
    Args:
        proof: The LEAN4 proof code to verify
        kimina_server_url: URL of the Kimina server
        api_key: Optional API key for Kimina server
        
    Returns:
        True if proof is correct (no errors), False otherwise
    """
    try:
        client = AsyncKiminaClient(kimina_server_url, api_key=api_key)
        result = await client.check(proof)
        
        # Check if result indicates success
        if isinstance(result, list) and len(result) > 0:
            repl_response = result[0]
            
            # Convert to dict if needed
            if not isinstance(repl_response, dict):
                if hasattr(repl_response, "model_dump"):
                    repl_response = repl_response.model_dump()
                elif hasattr(repl_response, "__dict__"):
                    repl_response = repl_response.__dict__
                else:
                    return False
            
            response = repl_response.get("response", {})
            if not isinstance(response, dict):
                if hasattr(response, "model_dump"):
                    response = response.model_dump()
                elif hasattr(response, "__dict__"):
                    response = response.__dict__
                else:
                    return False
            
            messages = response.get("messages", [])
            
            # Check for errors - if any message has severity "error", proof is incorrect
            for msg in messages:
                if not isinstance(msg, dict):
                    if hasattr(msg, "model_dump"):
                        msg = msg.model_dump()
                    elif hasattr(msg, "__dict__"):
                        msg = msg.__dict__
                    else:
                        continue
                
                severity = msg.get("severity", "").lower()
                if severity == "error":
                    return False
            
            # If no errors found, proof is correct
            return True
        else:
            # Empty or unexpected result format
            return False
            
    except Exception as e:
        # Any exception means verification failed
        print(f"Error verifying LEAN4 proof: {e}")
        return False


def _extract_lean_proof(solution_str: str) -> str:
    """
    Extract LEAN4 proof code from the solution string.
    The model might output the proof in various formats (with markdown, explanations, etc.)
    
    Args:
        solution_str: The full solution string from the model
        
    Returns:
        Extracted LEAN4 code
    """
    # Try to extract code blocks first
    code_block_pattern = r"```(?:lean|lean4)?\s*\n(.*?)```"
    matches = re.findall(code_block_pattern, solution_str, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Take the last code block
    
    # If no code blocks, try to find LEAN4 statements (theorem, def, etc.)
    # This is a simple heuristic - you may need to adjust based on your model's output format
    lean_pattern = r"(?:theorem|def|lemma|example|#check|#eval).*?(?=\n\n|\Z)"
    matches = re.findall(lean_pattern, solution_str, re.DOTALL)
    if matches:
        return "\n".join(matches).strip()
    
    # If no pattern matches, return the whole string (might be pure LEAN4 code)
    return solution_str.strip()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Optional[str] = None,  # Not used for LEAN4, but required by signature
    extra_info: Optional[dict] = None,
    kimina_server_url: str = "http://localhost:8000",
    kimina_api_key: Optional[str] = None,
    **kwargs
) -> float:
    """
    Compute reward score for LEAN4 proof verification.
    
    Args:
        data_source: Dataset source identifier
        solution_str: The model's solution/proof string
        ground_truth: Not used for LEAN4 (proofs are verified, not compared)
        extra_info: Optional extra information dict (may contain kimina_server_url, etc.)
        kimina_server_url: URL of Kimina server (can be overridden in extra_info)
        kimina_api_key: Optional API key for Kimina server
        **kwargs: Additional keyword arguments
        
    Returns:
        1.0 if proof is correct, 0.0 otherwise
    """
    # Get Kimina server URL from extra_info if provided
    if extra_info:
        kimina_server_url = extra_info.get("kimina_server_url", kimina_server_url)
        kimina_api_key = extra_info.get("kimina_api_key", kimina_api_key)
    
    # Extract LEAN4 proof from solution string
    proof = _extract_lean_proof(solution_str)
    
    if not proof:
        return 0.0
    
    # Verify proof using Kimina (synchronous wrapper for async function)
    try:
        # Check if we're in an async context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to use a different approach
                # For now, create a new event loop in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        _verify_lean_proof_async(proof, kimina_server_url, kimina_api_key)
                    )
                    is_correct = future.result(timeout=30)  # 30 second timeout
            else:
                is_correct = loop.run_until_complete(
                    _verify_lean_proof_async(proof, kimina_server_url, kimina_api_key)
                )
        except RuntimeError:
            # No event loop exists, create a new one
            is_correct = asyncio.run(
                _verify_lean_proof_async(proof, kimina_server_url, kimina_api_key)
            )
    except Exception as e:
        print(f"Error in compute_score for LEAN4: {e}")
        return 0.0
    
    return 1.0 if is_correct else 0.0

