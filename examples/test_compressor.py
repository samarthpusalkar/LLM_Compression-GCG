import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from llm_compressor import LLMCompressor

def generate_sample_texts():
    texts = {}
    
    # 1. Short Text (~30 tokens)
    texts["short.txt"] = "The quick brown fox jumps over the lazy dog. AI compression engines utilize Shannon entropy limits to perfectly reconstruct semantic structures."
    
    # 2. Medium Text (~300 tokens)
    texts["medium.txt"] = (
        "In the year 2045, humanity reached a technological singularity. " * 10 +
        "Artificial Intelligence systems began optimizing their own codebases, creating " +
        "recursive feedback loops of intelligence explosion. " * 5 +
        "Eventually, data storage became an obsolete concept. Instead of storing actual files, " +
        "quantum servers simply stored the initial mathematical seeds and the deterministic " +
        "divergence coordinates. When a user requested a file, the AI would re-hallucinate " +
        "the exact document perfectly on the fly, rendering traditional hard drives obsolete."
    )
    
    # 3. Long Text (~1500 tokens)
    # We repeat paragraphs to simulate a longer document.
    paragraph = (
        "The fundamental theorem of calculus connects differentiation and integration. "
        "It states that if a function is continuous on an interval, its definite integral "
        "can be evaluated using its antiderivative. This revolutionary concept, independently "
        "developed by Isaac Newton and Gottfried Wilhelm Leibniz in the 17th century, "
        "formed the backbone of modern physics and engineering. Without it, the precise "
        "calculations required for orbital mechanics, fluid dynamics, and electromagnetic "
        "theory would be impossible. "
    )
    texts["long.txt"] = paragraph * 30  # Roughly 1500-2000 tokens

    # Write them to disk
    os.makedirs("test_files", exist_ok=True)
    for filename, content in texts.items():
        filepath = os.path.join("test_files", filename)
        with open(filepath, "w") as f:
            f.write(content)
            
    return list(texts.keys())

def run_tests():
    compressor = LLMCompressor(model_name="Qwen/Qwen2.5-0.5B")
    
    filenames = generate_sample_texts()
    
    print("\n" + "="*60)
    print("STARTING LLM COMPRESSION ENGINE TESTS")
    print("="*60 + "\n")
    
    for filename in filenames:
        filepath = os.path.join("test_files", filename)
        db_path = os.path.join("test_files", f"{filename}.db.json")
        
        with open(filepath, "r") as f:
            original_text = f.read()
            
        print(f"Testing File: {filename}")
        
        # --- COMPRESSION ---
        start_time = time.time()
        total_toks, surprise_toks = compressor.compress(original_text, db_path)
        comp_time = time.time() - start_time
        
        compression_ratio = ((total_toks - surprise_toks) / total_toks) * 100
        print(f"  [COMPRESS] Time: {comp_time:.2f}s | Tokens: {total_toks} | Surprises: {surprise_toks} | Ratio: {compression_ratio:.1f}% compressible")
        
        # --- DECOMPRESSION ---
        start_time = time.time()
        decompressed_text = compressor.decompress(db_path)
        decomp_time = time.time() - start_time
        
        print(f"  [DECOMPRESS] Time: {decomp_time:.2f}s")
        
        # --- VERIFICATION ---
        if original_text == decompressed_text:
            print(f"  [VERDICT] ✅ SUCCESS! Lossless reconstruction verified.")
        else:
            print(f"  [VERDICT] ❌ FAILED! Output does not match original.")
            print(f"  Original snippet: {repr(original_text[:50])}...")
            print(f"  Decompressed snippet: {repr(decompressed_text[:50])}...")
            
        print("-" * 60)

if __name__ == "__main__":
    run_tests()
