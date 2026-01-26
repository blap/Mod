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

print("Carregando modelo com offloading para economizar memória...")
# Configurar offloading para usar disco como memória adicional
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,  # Usar float16 para economizar memória
    device_map="auto",  # Mapeamento automático para usar CPU e GPU conforme necessário
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    offload_folder="./offload",  # Pasta para offloading
    offload_state_dict=True,  # Fazer offload do estado do modelo
    max_memory={0: "4GiB", "cpu": "32GiB"}  # Limitar uso de memória
)

print(f"Modelo carregado com sucesso!")
print(f"Dispositivo do modelo: {next(model.parameters()).device}")
print(f"Número de parâmetros: {model.num_parameters() / 1e9:.2f}B")

print("Movendo entradas para o dispositivo do modelo...")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

print("Gerando resposta...")
generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)

print("Decodificando resposta...")
output_text = tokenizer.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("Resultado:")
print(output_text)