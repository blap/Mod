"""
Script para atualizar todos os arquivos de configuração para seguir o padrão de configuração autocontida.

Este script atualiza os arquivos de configuração existentes para:
1. Garantir que cada modelo tenha sua classe específica de configuração
2. Incluir lógica para detectar e usar caminho do drive H quando disponível
3. Manter valores padrão específicos para o modelo dentro do arquivo de configuração
4. Permitir sobreposição de configurações específicas do modelo
"""

import os
import re
from pathlib import Path


def update_config_files():
    """Atualiza todos os arquivos de configuração para seguir o padrão autocontido."""
    
    # Diretório raiz do projeto
    root_dir = Path("C:/Users/Admin/Documents/GitHub/Mod")
    
    # Encontrar todos os arquivos config.py
    config_files = list(root_dir.rglob("**/config.py"))
    
    print(f"Encontrados {len(config_files)} arquivos de configuração para atualizar:")
    for config_file in config_files:
        print(f"  - {config_file}")
    
    for config_file in config_files:
        print(f"\nAtualizando: {config_file}")
        
        # Ler o conteúdo do arquivo
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verificar se já tem a lógica de detecção do drive H
        if '"H:/' in content or "'H:/" in content:
            print(f"  Arquivo já contém lógica de detecção do drive H")
            continue
        
        # Extrair o nome do modelo a partir do caminho
        model_name = extract_model_name(config_file)
        
        # Atualizar o conteúdo com a lógica de detecção do drive H
        updated_content = add_h_drive_logic(content, model_name)
        
        # Escrever o conteúdo atualizado de volta ao arquivo
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"  Arquivo atualizado com sucesso!")


def extract_model_name(config_file_path):
    """Extrai o nome do modelo a partir do caminho do arquivo."""
    path_parts = str(config_file_path).split(os.sep)
    
    # Procurar por pastas de modelo no caminho
    for part in path_parts:
        if 'qwen3_' in part or 'coder' in part or 'vl' in part or 'glm' in part:
            return part.replace('\\', '').replace('/', '')
    
    # Se não encontrar um nome específico, tentar extrair do nome do arquivo ou comentário
    with open(config_file_path, 'r', encoding='utf-8') as f:
        first_lines = f.read(500)  # Ler primeiras 500 linhas para encontrar nome
    
    # Procurar por nomes de modelo em comentários ou docstrings
    import re
    matches = re.findall(r'(qwen3[_\-][^"\s]+)', first_lines, re.IGNORECASE)
    if matches:
        return matches[0].replace('-', '_')
    
    return "modelo_generico"


def add_h_drive_logic(content, model_name):
    """Adiciona a lógica de detecção do drive H ao conteúdo do arquivo."""
    
    # Converter o nome do modelo para o formato esperado
    h_drive_path = f"H:/{model_name.replace('_', '-')}".replace('--', '-').title().replace('-', '')
    
    # Verificar se já existe uma chamada para get_default_model_path
    if 'get_default_model_path' in content:
        # Substituir a lógica existente para adicionar a verificação do drive H
        # Primeiro, encontrar o método __post_init__
        post_init_match = re.search(r'def __post_init__\(self\):(.*?)(?=^\s*def|\Z)', content, re.DOTALL | re.MULTILINE)
        
        if post_init_match:
            post_init_content = post_init_match.group(1)
            
            # Verificar se já tem a lógica de detecção do drive H
            if 'H:/' not in post_init_content:
                # Modificar o conteúdo do __post_init__ para adicionar a lógica do drive H
                # Encontrar onde o model_path é definido
                if 'self.model_path = get_default_model_path' in post_init_content:
                    # Adicionar lógica após a chamada para get_default_model_path
                    new_post_init = post_init_content.replace(
                        'self.model_path = get_default_model_path',
                        f'# Ensure the model path points to the H drive for {model_name} model\n'
                        f'        if (\n'
                        f'            not self.model_path\n'
                        f'            or "{model_name.lower()}" in self.model_path.lower()\n'
                        f'        ):\n'
                        f'            self.model_path = "{h_drive_path}"\n\n'
                        f'        # Call parent\'s post_init to validate config\n'
                        f'        super().__post_init__()\n\n'
                        f'        self.model_path = get_default_model_path'
                    )
                    
                    # Substituir o conteúdo antigo pelo novo
                    content = content.replace(post_init_content, new_post_init)
                else:
                    # Se não encontrar a chamada para get_default_model_path, adicionar a lógica no início do __post_init__
                    new_post_init = (
                        f'        # Set default model path if not provided\n'
                        f'        if not self.model_path:\n'
                        f'            self.model_path = get_default_model_path(self.model_name)\n\n'
                        f'        # Ensure the model path points to the H drive for {model_name} model\n'
                        f'        if (\n'
                        f'            not self.model_path\n'
                        f'            or "{model_name.lower()}" in self.model_path.lower()\n'
                        f'        ):\n'
                        f'            self.model_path = "{h_drive_path}"\n\n'
                        f'        # Call parent\'s post_init to validate config\n'
                        f'        super().__post_init__()\n'
                        f'{post_init_content}'
                    )
                    
                    # Substituir o conteúdo antigo pelo novo
                    content = content.replace(post_init_match.group(0), 
                                             f'def __post_init__(self):{new_post_init}')
        else:
            # Se não encontrar __post_init__, adicionar o método
            # Encontrar onde termina a definição da classe
            class_end_match = re.search(r'(def __post_init__\(self\):|class \w+.*?:|@dataclass)', content)
            if not class_end_match:
                # Se não encontrar nenhum marcador claro, adicionar no final antes do registro
                if 'register_model_config' in content:
                    parts = content.split('register_model_config')
                    content = (parts[0] + 
                              f'\n    def __post_init__(self):\n'
                              f'        """Post-initialization adjustments."""\n'
                              f'        # Set default model path if not provided\n'
                              f'        if not self.model_path:\n'
                              f'            self.model_path = get_default_model_path(self.model_name)\n\n'
                              f'        # Ensure the model path points to the H drive for {model_name} model\n'
                              f'        if not self.model_path or "{model_name.lower()}" in self.model_path.lower():\n'
                              f'            self.model_path = "{h_drive_path}"\n\n'
                              f'        # Call parent\'s post_init to validate config\n'
                              f'        super().__post_init__()\n\n'
                              f'register_model_config' + 
                              parts[1])
                else:
                    # Adicionar no final
                    content += f'\n    def __post_init__(self):\n'
                    content += f'        """Post-initialization adjustments."""\n'
                    content += f'        # Set default model path if not provided\n'
                    content += f'        if not self.model_path:\n'
                    content += f'            self.model_path = get_default_model_path(self.model_name)\n\n'
                    content += f'        # Ensure the model path points to the H drive for {model_name} model\n'
                    content += f'        if not self.model_path or "{model_name.lower()}" in self.model_path.lower():\n'
                    content += f'            self.model_path = "{h_drive_path}"\n\n'
                    content += f'        # Call parent\'s post_init to validate config\n'
                    content += f'        super().__post_init__()\n\n'
    
    return content


if __name__ == "__main__":
    update_config_files()
    print("\nAtualização concluída!")