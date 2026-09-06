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
  <a href="https://github.com/idletoken/IdleToken/releases">下载桌面客户端</a>
  · <a href="https://idletoken.ai">进入市场</a>
  · <a href="README.md">English</a>
</p>

## 为什么做 IdleToken

智能体对算力的需求有明显峰谷，家里的 GPU 却在大部分时间里处于闲置状态。IdleToken 让不同机器的算力彼此补位：闲时分享算力赚取火花，需要时再用火花调用别人分享的模型服务。

## 两种开始方式

### 部署并分享自己的模型

1. 在带 NVIDIA 显卡的 Windows / Linux 电脑，或 Apple Silicon Mac 上安装桌面客户端。
2. 选择精选模型、量化精度和上下文，在一台电脑或异构系统组成的局域网集群上运行。
3. 打开共享赚取火花；需要更多算力时打开外援，调用他人分享的资源。

### 调用别人分享的模型

1. 在 [idletoken.ai](https://idletoken.ai) 注册账号。
2. 进入**火花 → API 密钥**并创建密钥。
3. 接入任意兼容 Anthropic 或 OpenAI 的客户端。无需显卡，也无需安装桌面客户端。

新账号的火花余额为零。你可以通过分享算力赚取，或使用兑换码；目前暂不支持购买火花。

## 接入现有工具

| 用途 | Base URL | API key |
| --- | --- | --- |
| 自己的本地模型 | `http://127.0.0.1:8000` | 任意非空值 |
| 共享算力市场 | `https://api.idletoken.ai` | 在门户创建的密钥 |

Claude Code 示例：

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_API_KEY=idletoken
claude
```

OpenAI 兼容 API 使用同一个 Base URL。`GET /v1/models` 会返回当前 endpoint 可用的模型 ID。

## 你会得到什么

- **一套客户端，单机和集群都能用。** 单机直接调用 llama.cpp；集群可在同一局域网内混合使用 Windows、Linux 和 macOS 电脑并切分模型。
- **两套熟悉的 API。** 同时兼容 OpenAI 与 Anthropic，Claude Code 是核心使用场景。
- **本地优先的边界。** 本机 API 只监听 `127.0.0.1`；集群张量流量只走可直连的局域网，并使用 PSK-TLS 保护。
- **明确的资源选择。** 可选择 256K，或在模型支持时选择 1M 上下文；资源不足会明确拒绝，不会静默缩短窗口或改变部署方式。

## 模型

IdleToken 提供精选的 Qwen、OpenAI 与 DeepSeek GGUF 文本生成模型。版本化清单位于 [`models/`](models/)；客户端下载权重并完成完整性校验、资源估算和量化精度选择。

每个入列模型都可在装得下时单机运行，也可由用户选择集群部署。当前不支持多模态输入或任意本地 GGUF 文件。需要其它模型？欢迎[提交 issue](https://github.com/idletoken/IdleToken/issues)。

## 硬件

- Windows 10/11：NVIDIA GPU，计算能力不低于 7.5、显存至少 4 GB，并安装当前驱动；安装包已包含 CUDA 运行时 DLL。
- Linux：显卡要求相同；x86_64 需要不低于 570.26 的驱动与 CUDA 12.8，arm64 需要不低于 580.65 的驱动与 CUDA 13.0。
- macOS：Apple Silicon，统一内存足以容纳所选模型。
- 纯 CPU 电脑、AMD / Intel GPU、Intel Mac 与手机只能控制集群，不参与推理。
- 集群节点必须运行同一版本的 IdleToken，并能通过局域网直连；张量流量不走 Tailscale 等 VPN 或覆盖网络。

## 从源码构建

仓库包含 Tauri 客户端及其所需的全部原生 sidecar：coordinator、worker supervisor、platform agent，以及钉住版本的 llama.cpp server 与 RPC server。以下命令均从仓库根目录开始执行。

### 前置依赖

只调试前端需要 Node.js 与 pnpm。完整原生构建还需要 Git、CMake、Rust 1.77 或更新版本，以及目标操作系统对应的 [Tauri v2 系统依赖](https://tauri.app/start/prerequisites/)。

- **Linux：** C 编译器，以及“硬件”一节中对应架构的 CUDA Toolkit。
- **macOS：** Apple Silicon 与 Xcode Command Line Tools。
- **Windows：** Visual Studio 2022 Build Tools（勾选 **Desktop development with C++**）、Windows SDK、带 Visual Studio Integration 的 CUDA Toolkit 12.8，以及提供 `gcc` 和 `windres` 的 WinLibs/MinGW。若这些工具不在 `PATH`，请设置 `IDLETOKEN_MINGW_BIN`。

### 只调试前端

这种方式不会启动原生引擎；浏览器界面会使用明确标记的开发夹具。

```sh
cd client
pnpm install
pnpm dev
```

浏览器访问 `http://localhost:1420`。

### 在 Linux 或 macOS 上运行完整桌面开发版

```sh
./scripts/build_llamacpp.sh
make
make -f Makefile.platform
./scripts/stage_sidecars.sh
cd client
pnpm install
pnpm tauri dev
```

Linux 构建 CUDA 后端，macOS 构建 Metal 后端。缺少 GPU 工具链时会明确失败，不会退回 CPU。

### 构建可安装包

打包脚本会验证所有 sidecar 是否齐全，并确认安装包中的引擎与钉住的 llama.cpp revision 一致。Linux 与 macOS 发布打包要求 Git 工作树干净，使产物能够对应到一个精确的源码 commit。

Linux（`.deb` 与 `.rpm`）：

```sh
./scripts/build_llamacpp.sh
make IDLETOKEN_PLATFORM_VERIFY_KEY_B64="$(tr -d '\r\n' < scripts/platform-verify-key.b64)"
make -f Makefile.platform
./scripts/build_client_release.sh
```

产物位于 `client/src-tauri/target/release/bundle/`。

macOS（`.dmg`）：

```sh
./scripts/build_llamacpp.sh
./scripts/package_client_mac.sh
```

Windows（`.exe`，请在 x64 Native Tools Command Prompt 中执行）：

```bat
scripts\build_llamacpp_win.bat
cd client
pnpm install
pnpm build:release
cd ..
scripts\build_client_release.bat
```

Windows 安装包位于 `client\src-tauri\target\release\bundle\nsis\`。发布脚本会重新构建 coordinator、worker 与 platform agent，暂存钉住版本的 llama.cpp sidecar 和运行时 DLL，最后调用 NSIS bundler。

### 无需推理硬件的检查

```sh
scripts/acceptance.sh --gate G_MODEL
python3 scripts/model_manifest_check.py
cd client && npx tsc --noEmit
```

完整验收阶梯由 [`scripts/acceptance.sh`](scripts/acceptance.sh) 实现。需要硬件的门可参考 [`scripts/testbed.env.example`](scripts/testbed.env.example) 配置自己的机器。

## 获取帮助

遇到安装、组网或 API 问题时，请先搜索[已有 issue](https://github.com/idletoken/IdleToken/issues)；若需新建 issue，请附上操作系统、IdleToken 版本、硬件信息与相关报错或日志片段。

## 项目链接

[版本发布](https://github.com/idletoken/IdleToken/releases) · [问题反馈](https://github.com/idletoken/IdleToken/issues) · [安全政策](SECURITY.md) · [参与贡献](CONTRIBUTING.md) · [Apache-2.0 许可](LICENSE)

项目基于 [llama.cpp](https://github.com/ggml-org/llama.cpp) 与 [Tauri](https://github.com/tauri-apps/tauri) 构建；第三方项目致谢见 [NOTICE](NOTICE)。
