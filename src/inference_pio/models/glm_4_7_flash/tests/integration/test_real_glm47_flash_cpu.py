import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Usando o caminho local do modelo GLM-4.7-Flash
MODEL_PATH = "H:/GLM-4.7-Flash"

# Mensagem de teste
messages = [{"role": "user", "content": "hello"}]

print("Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Aplicando template de chat...")
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

print("Carregando modelo com otimizações de memória...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,  # Usar float16 para economizar memória
    device_map="cpu",  # Usar CPU para evitar problemas de memória GPU
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

print(f"Modelo carregado com sucesso!")
print(f"Dispositivo do modelo: {next(model.parameters()).device}")
print(f"Número de parâmetros: {model.num_parameters() / 1e9:.2f}B")

print("Movendo entradas para o dispositivo do modelo...")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

print("Gerando resposta...")
generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

print("Decodificando resposta...")
output_text = tokenizer.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("Resultado:")
print(output_text)