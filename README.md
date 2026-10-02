<div align="center">

# Facial Symmetry Analysis

**Sistema de análise de simetria e expressão facial em tempo real utilizando visão computacional e câmeras OAK-D.**

<br>

<img src="https://skillicons.dev/icons?i=python,opencv,tensorflow,sklearn,numpy,pandas" />

<br><br>

<img src="https://skillicons.dev/icons?i=pycharm,vscode,github" />

</div>

---

## Sobre o Projeto

**Facial Symmetry Analysis** é uma aplicação desenvolvida em Python para detecção facial, análise de simetria e reconhecimento de expressões faciais em tempo real.

O projeto utiliza técnicas de **Computer Vision, Machine Learning e Deep Learning** para processar imagens capturadas por câmeras OAK-D da **Luxonis**, permitindo identificar características faciais, gerar uma malha facial e analisar relações geométricas entre diferentes regiões do rosto.

A aplicação possui uma interface gráfica desenvolvida com **PySimpleGUI** e também pode ser distribuída como um executável standalone para Windows utilizando **PyInstaller**.

## Tecnologias

<div align="center">

<img src="https://skillicons.dev/icons?i=python,opencv,tensorflow,sklearn,numpy,pandas,matplotlib" />

</div>

### Computer Vision

* **OpenCV** — Processamento e análise de imagens
* **MediaPipe** — Detecção e mapeamento de landmarks faciais
* **DepthAI** — Integração com câmeras OAK-D
* **OpenVINO** — Inferência de modelos de visão computacional
* **blobconverter** — Conversão e preparação de modelos para dispositivos compatíveis

### Machine Learning

* **Keras** — Construção e execução de modelos de Deep Learning
* **Scikit-learn** — Processamento e análise de dados
* **FER** — Facial Expression Recognition

### Dados e processamento

* **NumPy** — Computação numérica e operações vetoriais
* **Pandas** — Manipulação e análise de dados
* **Matplotlib** — Visualização de dados

### Interface e distribuição

* **PySimpleGUI** — Interface gráfica
* **PyInstaller** — Empacotamento da aplicação em executável standalone

## Funcionalidades

### Captura em tempo real

Integração com câmeras **OAK-D da Luxonis** através da biblioteca DepthAI para captura e processamento de imagens em tempo real.

### Detecção facial

Identificação de rostos utilizando modelos de visão computacional e processamento através de OpenVINO.

### Facial Landmark Detection

Mapeamento de pontos de referência faciais utilizando MediaPipe para identificar regiões como:

```text
Olhos
Sobrancelhas
Nariz
Boca
Mandíbula
Contorno facial
```

### Análise de simetria

A aplicação utiliza os landmarks faciais para criar relações geométricas entre os dois lados do rosto.

O sistema pode analisar:

* Eixo central facial
* Distância entre landmarks
* Relações geométricas
* Diferenças entre regiões correspondentes
* Simetria relativa entre os lados do rosto
* Malha facial 3D

### Reconhecimento de expressões

Utilização de modelos de reconhecimento facial para classificação de expressões detectadas durante a captura de vídeo.

### Análise de dados

Os dados processados podem ser manipulados e analisados utilizando:

```text
NumPy
Pandas
Scikit-learn
Matplotlib
```

### Interface gráfica

Interface desenvolvida em **PySimpleGUI**, permitindo visualizar o processamento da câmera, landmarks faciais e informações relacionadas à análise.

### Executável standalone

A aplicação pode ser empacotada utilizando **PyInstaller**, permitindo executar o sistema em máquinas Windows sem a necessidade de configurar manualmente o ambiente Python.

## Arquitetura

```text
                    ┌─────────────────────┐
                    │      OAK-D Camera   │
                    │       Luxonis       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      DepthAI        │
                    │   Camera Pipeline   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Face Detection   │
                    │      OpenVINO       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Facial Landmarks  │
                    │     MediaPipe       │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │ Symmetry        │         │ Expression      │
        │ Analysis        │         │ Recognition     │
        └────────┬────────┘         └────────┬────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                    ┌─────────────────────┐
                    │     GUI / Output     │
                    │     PySimpleGUI      │
                    └─────────────────────┘
```

## Estrutura do Projeto

```text
FacialSymmetryAnalysis/
│
├── Python_scripts_and_spec_file/
│   ├── FacialExpressionAnalysis.py
│   └── FacialExpressionAnalysis.spec
│
├── for_conversion/
│   └── requirements.txt
│
├── standalone_executable/
│   └── FacialSymmetryAnalysis.exe
│
└── README.md
```

### Principais diretórios

**`Python_scripts_and_spec_file/`**

Contém o código-fonte principal da aplicação e os arquivos de configuração utilizados pelo PyInstaller.

**`for_conversion/`**

Contém as dependências necessárias para configurar o ambiente de desenvolvimento.

**`standalone_executable/`**

Contém a versão compilada da aplicação para Windows.

## Instalação

### Requisitos

Para executar a versão de desenvolvimento, recomenda-se:

```text
Python 3.x
Windows
Câmera OAK-D
USB 3.0
```

### Clone o repositório

```bash
git clone https://github.com/SEU_USUARIO/FacialSymmetryAnalysis.git
cd FacialSymmetryAnalysis
```

### Instale as dependências

```bash
cd for_conversion
pip install -r requirements.txt
```

### Execute a aplicação

Após configurar as dependências e conectar a câmera OAK-D, execute o script principal:

```bash
python FacialExpressionAnalysis.py
```

## Executável Windows

Para utilizar a versão standalone:

```text
standalone_executable/
└── FacialSymmetryAnalysis.exe
```

Execute o arquivo `.exe` após conectar a câmera OAK-D.

A versão standalone foi desenvolvida para reduzir a necessidade de configuração manual do ambiente Python na máquina final.

## Build

Caso sejam realizadas alterações no código-fonte, um novo executável pode ser criado utilizando PyInstaller.

```bash
cd Python_scripts_and_spec_file
pyinstaller FacialSymmetryAnalysis.spec
```

Antes do processo de build, certifique-se de que todos os caminhos utilizados no código e no arquivo `.spec` estejam configurados de acordo com o ambiente atual.

Evite utilizar caminhos absolutos específicos de uma máquina.

Exemplo:

```text
D:\Desktop\Projeto\Modelos\model.xml
```

Prefira caminhos relativos ou configuráveis:

```text
models/model.xml
```

## Configuração

### Hardware

O projeto foi desenvolvido originalmente para utilização com câmeras:

**Luxonis OAK-D**

A câmera é utilizada para captura de imagens e processamento através da pipeline do DepthAI.

### Modelos

O sistema utiliza modelos de visão computacional para detecção facial e processamento dos frames.

Os modelos devem estar corretamente configurados antes da execução.

### Dependências

As principais dependências podem ser encontradas em:

```text
for_conversion/requirements.txt
```

## Fluxo de Processamento

```text
Camera
   ↓
Frame Capture
   ↓
Face Detection
   ↓
Facial Landmarks
   ↓
Geometric Analysis
   ↓
Symmetry Calculation
   ↓
Expression Recognition
   ↓
Data Processing
   ↓
Visualization
```

## Screenshots

<div align="center">

Adicione aqui capturas da aplicação em funcionamento.

<br>

Exemplos:

* Interface principal
* Detecção facial
* Facial landmarks
* Malha facial
* Análise de simetria
* Reconhecimento de expressões
* Gráficos e métricas

</div>

## Roadmap

* [ ] Melhorar o gerenciamento de diretórios e arquivos de modelos
* [ ] Remover caminhos absolutos do código-fonte
* [ ] Adicionar suporte para webcams convencionais
* [ ] Implementar testes automatizados
* [ ] Melhorar a análise geométrica facial
* [ ] Implementar métricas detalhadas de simetria
* [ ] Adicionar armazenamento histórico das análises
* [ ] Criar dashboard para visualização dos resultados
* [ ] Melhorar o processamento em tempo real
* [ ] Otimizar inferência para diferentes hardwares
* [ ] Implementar configuração dinâmica dos modelos

## Desenvolvimento

O projeto foi desenvolvido utilizando conceitos de:

```text
Computer Vision
Facial Landmark Detection
Machine Learning
Deep Learning
Real-Time Image Processing
Geometric Analysis
Data Analysis
Edge AI
```

## Autor

Projeto desenvolvido para pesquisa e experimentação em **visão computacional, análise facial e inteligência artificial**.

Contribuições, issues e pull requests são bem-vindos.

---

<div align="center">

**Facial Symmetry Analysis**

Computer Vision • Machine Learning • Edge AI

</div>
