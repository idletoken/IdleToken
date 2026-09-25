<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.svg">
    <img src="docs/images/logo.svg" alt="IdleToken" width="380">
  </picture>
</p>

<h1 align="center">分享你的空闲 Token。</h1>

<p align="center">
  本地运行大模型，闲时分享给别人，忙时调用别人的。
</p>

<p align="center">
  <a href="https://github.com/idletoken/IdleToken/releases">⬇️ 下载客户端</a>
  · <a href="https://idletoken.ai">🏠 项目主页</a>
  · <a href="README.md">🌐 English</a>
</p>

---

## ✨ 为什么做 IdleToken

智能体对算力的需求天然有峰谷。平时只有一两个推理任务在运行；遇到复杂任务时，多个智能体同时工作，短时间内发出大量推理请求。

设备的使用也有自己的峰谷：用户离开或夜间不用时闲置吃灰，用户需要时又经常不够用（如同时开启多个应用还要部署大模型）。

IdleToken 想做的，就是把这些错开的峰谷连接起来：让闲置的机器，为正处于算力高峰的智能体提供算力。

<table>
<tr>
<td width="50%" align="center">
<img src="docs/images/why-idle.svg" alt="空闲 GPU 分享富余算力" width="320"><br>
<strong>🌿 机器空闲时</strong><br>
<sub>把富余算力分享给别人。</sub>
</td>
<td width="50%" align="center">
<img src="docs/images/why-busy.svg" alt="繁忙机器请求其他机器协助" width="320"><br>
<strong>⚡ 机器繁忙时</strong><br>
<sub>请求别人完成排队中的任务。</sub>
</td>
</tr>
</table>

我们使用“火花”来结算 Token。

## 💬 加入社区

加入 IdleToken Discord，交流使用与开发。

[![Discord](https://img.shields.io/badge/Discord-%E5%8A%A0%E5%85%A5%E7%A4%BE%E5%8C%BA-5865F2?logo=discord&logoColor=white)](https://discord.gg/XbaWCH4t2J)

## 🚀 两种开始方式

> 📖 **第一次用？[手把手使用教程](docs/user-guide.zh-CN.md)配了截图，从头走一遍**——
> 安装、注册、跑通第一个模型、多机组网、分享算力、接 Claude Code。

### 🖥️ 部署并分享自己的模型

1. [下载并安装客户端](https://github.com/idletoken/IdleToken/releases)。
2. 启动客户端，下载并部署模型。
3. 打开“外援”按钮，请求别人完成本地排队的任务。
4. 打开“共享”按钮，帮助别人完成任务。

### ☁️ 调用别人分享的模型

如果你不想自己部署，例如希望在手机、平板或没有受支持 GPU 的电脑上使用 IdleToken，可以直接调用别人分享的模型。

1. 在 [idletoken.ai](https://idletoken.ai/register) 注册账号。
2. 在账户中创建 API 密钥。
3. 使用 API 调用别人分享的模型。

## 🔌 接入第三方

- **连接自己部署的模型：** `http://127.0.0.1:8000`。模型启动后，即可通过这个本机地址接入。
- **调用别人分享的模型：** `https://api.idletoken.ai`。登录后，点击右上角用户头像，进入「火花」→「API 密钥」，点击「新建密钥」。

### 接入 DeepSeek Harness

[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)（`dsh`）通过自定义服务商接入 IdleToken。在 `$DSH_HOME/settings.yaml`（默认为 `~/.dsh/settings.yaml`）中添加以下配置，将 `MODEL_ID` 替换为 `GET /v1/models` 返回的模型 ID：

```yaml
llm-pi-ai:
  providers:
    idletoken:
      api: openai-completions
      baseURL: https://api.idletoken.ai/v1
      apiKeyEnv: IDLETOKEN_API_KEY
      models:
        - id: MODEL_ID
```

```sh
export IDLETOKEN_API_KEY='sk-idletoken-********************************'
npx @deepseek-ai/dsh web
```

Web 界面默认打开 `http://127.0.0.1:3080`，在 `idletoken` 服务商下选择模型即可。手动添加的
模型默认按纯文本处理，视觉模型需要在其 `id` 旁补一行 `input: [text, image]`。

### 接入 OpenCode

在项目根目录创建 `opencode.json`，将 `MODEL_ID` 替换为 `GET /v1/models` 返回的模型 ID：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "idletoken/MODEL_ID",
  "provider": {
    "idletoken": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "IdleToken",
      "options": {
        "baseURL": "https://api.idletoken.ai/v1",
        "apiKey": "{env:IDLETOKEN_API_KEY}"
      },
      "models": {
        "MODEL_ID": {
          "name": "IdleToken"
        }
      }
    }
  }
}
```

```sh
export IDLETOKEN_API_KEY='sk-idletoken-********************************'
opencode
```

## 🧠 模型

[当前支持的模型列表](models/)

需要其它模型？欢迎[提交 issue](https://github.com/idletoken/IdleToken/issues)。

## 💻 硬件

在本机部署模型的最低配置：

| 平台 | 最低配置 |
| --- | --- |
| **Windows 10/11 · Linux x86_64/arm64** | NVIDIA 显卡，RTX 20 系（Turing，计算能力 7.5）及以后，显存 8 GB，并安装当前 NVIDIA 驱动。已发布的 v0.1.90 Linux 二进制要求 glibc 2.39；当前源码构建以 glibc 2.35 为目标并在包元数据中声明。 |
| **macOS** | Apple Silicon Mac，统一内存 16 GB。 |

## 🖧 联机

我们的客户端支持将多台设备联机组网部署大模型，方法如下：

1. 在所有参与计算的机器上安装同一版本的 IdleToken，并确保它们可以通过局域网互相直连。
2. 每台机器选择相同的模型、量化精度和上下文，并完成模型参数下载。
3. 在一台机器上创建集群，其余机器登录同一账号或使用验证码加入。
4. 所有节点显示就绪后，由创建者选择联机并启动。IdleToken 会探测各节点资源并自动分配模型层。

---

## 🛠️ 从源码构建

### 前置依赖

- Git、CMake、Node.js 18+、pnpm、Rust 1.77+
- [Tauri v2 系统依赖](https://tauri.app/start/prerequisites/)
- Linux：C 编译器、对应架构的 CUDA Toolkit、pkg-config、libcurl 开发包（7.68+，带 TLS 与异步 DNS）、GLib 2.72+、Duktape 2.7 和 CA 证书
- macOS：Apple Silicon Mac 与 Xcode Command Line Tools
- Windows：Visual Studio 2022 Build Tools（Desktop development with C++）、Windows SDK、CUDA Toolkit 12.8（含 Visual Studio Integration）以及提供 `gcc`、`windres` 的 WinLibs/MinGW

桌面客户端的账户操作、平台 agent 与 coordinator overflow 共用同一套传输。公网 API 连接保留 HTTPS，并遵循机器的 HTTP/SOCKS 代理及 PAC 配置。Windows 构建会下载固定版本的静态 curl，使用 Windows 证书库；macOS 使用系统 libcurl。代理失败会明确报错，不会静默改为直连；本机推理绕过代理。Linux 安装包自带固定版本的私有 libproxy，不要求升级发行版的同名库。源码构建可用 `scripts/build_platform_proxy_linux.sh` 准备同一解析器；Debian/Ubuntu 构建依赖包括 `libcurl4-openssl-dev`、`libglib2.0-dev`、`duktape-dev`、`gsettings-desktop-schemas-dev`、`pkg-config`、`meson`、`ninja-build`、`patchelf`、`curl` 与 `ca-certificates`。原生代理解析放在有时间上限的子进程中，PAC 卡死不会永久占满调用端资源。设置环境变量 `CURL_CA_BUNDLE` 会覆盖所有平台连接使用的证书库；留空则使用系统证书库。

### 获取源码

```sh
git clone https://github.com/idletoken/IdleToken.git
cd IdleToken
```

### Linux

```sh
./scripts/build_llamacpp.sh
make all IDLETOKEN_PLATFORM_VERIFY_KEY_B64="$(tr -d '\r\n' < scripts/platform-verify-key.b64)"
make -f Makefile.platform
./scripts/build_client_release.sh
```

安装包位于 `client/src-tauri/target/release/bundle/deb/` 和
`client/src-tauri/target/release/bundle/rpm/`。其他 Linux 发行版可以基于开源代码自行适配，
但这些包不属于官方发布资产。

### macOS

```sh
./scripts/build_llamacpp.sh
./scripts/package_client_mac.sh
```

安装包位于 `client/src-tauri/target/release/bundle/dmg/`。

### Windows

在 x64 Native Tools Command Prompt 中执行：

```bat
scripts\build_llamacpp_win.bat
cd client
pnpm install
pnpm build:release
cd ..
scripts\build_client_release.bat
```

安装包位于 `client\src-tauri\target\release\bundle\nsis\`。

若构建工具不在默认位置，可设置 `IDLETOKEN_MINGW_BIN`、`IDLETOKEN_CUDA_RUNTIME_DIR` 或 `IDLETOKEN_MBEDTLS_SRC`。

## 💬 获取帮助

遇到安装、组网、API 或源码构建问题时，请先搜索[已有 issue](https://github.com/idletoken/IdleToken/issues)；若需新建 issue，请附上操作系统、IdleToken 版本、硬件信息、失败命令与相关报错或日志片段。

---

<p align="center">
  <a href="https://idletoken.ai">项目主页</a>
  · <a href="https://github.com/idletoken/IdleToken/releases">版本发布</a>
  · <a href="https://github.com/idletoken/IdleToken/issues">问题反馈</a>
  · <a href="LICENSE">Apache-2.0 许可</a>
</p>

<p align="center"><sub>第三方项目致谢见 <a href="NOTICE">NOTICE</a>。</sub></p>
