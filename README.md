# ✨ LUX Creative 969

> A production-ready transformer inference engine with a mobile-friendly dashboard — built for real-time AI model interaction, right from your browser.

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/Dashboard-Live-brightgreen)](https://smgibson19-prog.github.io/Lux-creative-969/)
[![Repository](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/smgibson19-prog/Lux-creative-969)

---

## 📖 What Is This?

**LUX Creative 969** is a JavaScript-based transformer inference engine combined with a mobile-friendly web dashboard. It lets you load and run AI language models directly in the browser — no server needed.

### Key Features

- 🤖 **Transformer inference engine** — full model loading and text generation in JavaScript
- 📱 **Mobile Dashboard** — responsive UI hosted on GitHub Pages
- 🚀 **Production ready** — auto-deploy via GitHub Actions
- 📊 **Real-time metrics** — monitor model performance live
- ♿ **Accessible design** — high contrast, clear navigation, screen-reader friendly

---

## ⚡ Quick Start

### Access the Mobile Dashboard (No Installation Needed)

1. Open your browser
2. Go to: **[https://smgibson19-prog.github.io/Lux-creative-969/](https://smgibson19-prog.github.io/Lux-creative-969/)**
3. The dashboard loads instantly — works on phones, tablets, and desktops

### Access the Live Web App

1. Visit the GitHub Pages link above
2. Use the navigation to explore features
3. No login or account required

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🤖 Transformer Engine | `model_runtime.js` handles model config, weight loading, and token generation |
| 📱 Mobile Dashboard | Hosted on GitHub Pages — works on any device |
| ☁️ Production Deployment | Auto-deploys to Azure via GitHub Actions on every push |
| 📊 Real-Time Metrics | Live performance indicators during model inference |
| 🎨 Responsive Design | Adapts to any screen size — phone, tablet, or desktop |
| ♿ Accessible UI | High contrast colors, clear labels, keyboard navigable |

---

## 🛠️ Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) v20 or later (for local development)
- A modern browser (Chrome, Firefox, Safari, Edge)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/smgibson19-prog/Lux-creative-969.git

# 2. Enter the project directory
cd Lux-creative-969

# 3. Install dependencies
npm install
```

### Running Locally

```bash
# Start a local development server
npm start
```

Then open your browser and go to **[http://localhost:3000](http://localhost:3000)**.

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `PORT` | Port for the local server | `3000` |
| `NODE_ENV` | Environment mode (`development` / `production`) | `development` |

Create a `.env` file in the project root to set these:

```env
PORT=3000
NODE_ENV=development
```

---

## 📁 Project Structure

```
Lux-creative-969/
│
├── model_runtime.js          # 🤖 Transformer inference engine
│                             #    Loads model config, weights, and runs generation
│
├── docs/                     # 📱 Mobile Dashboard (GitHub Pages)
│   ├── index.html            #    Dashboard homepage
│   ├── style.css             #    Styling and responsive layout
│   └── dashboard.js          #    Interactive functionality
│
├── index.html                # 🌐 Web app entry point
├── app.js                    # ⚙️  App initialization and routing
├── package.json              # 📦 Node.js dependencies and scripts
│
├── .github/
│   └── workflows/
│       ├── azure-webapps-node.yml   # ☁️  Azure deployment CI/CD
│       └── blank.yml                #    Additional workflow
│
├── LICENSE                   # 📄 BSD 3-Clause License
└── README.md                 # 📖 This file
```

---

## 🔗 Links

| Resource | URL |
|---|---|
| 📱 Live Dashboard | [https://smgibson19-prog.github.io/Lux-creative-969/](https://smgibson19-prog.github.io/Lux-creative-969/) |
| 💻 Repository | [https://github.com/smgibson19-prog/Lux-creative-969](https://github.com/smgibson19-prog/Lux-creative-969) |
| 🐛 Issues | [https://github.com/smgibson19-prog/Lux-creative-969/issues](https://github.com/smgibson19-prog/Lux-creative-969/issues) |

---

## ♿ Accessibility & Usability

This project is designed with accessibility in mind:

- 🎨 **High contrast colors** — easy to read in all lighting conditions
- 🔤 **Clear, large text** — readable on small mobile screens
- 🧭 **Simple navigation** — straightforward menus, no clutter
- ✅ **Status indicators** — plain language labels (e.g. "Ready", "Loading", "Error")
- ⌨️ **Keyboard navigable** — full keyboard support throughout the UI
- 📢 **Screen reader friendly** — proper HTML semantics and ARIA labels

---

## 📄 License

This project is licensed under the **BSD 3-Clause License**.

```
BSD 3-Clause License

Copyright (c) 2026, smgibson19-prog

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
```

See the [LICENSE](LICENSE) file for the full text.

---

## 🙋 Support & Contact

### 🐛 Found a Bug?

1. Go to the [Issues page](https://github.com/smgibson19-prog/Lux-creative-969/issues)
2. Click **"New issue"**
3. Describe what happened and what you expected
4. Include your browser and device type if relevant

### 💡 Want to Contribute?

1. **Fork** this repository
2. **Create a branch** for your change (`git checkout -b my-feature`)
3. **Make your changes** and commit them
4. **Open a pull request** describing what you changed and why

All contributions are welcome — bug fixes, improvements, documentation, and more! 💙

### 📬 Contact

- **GitHub:** [@smgibson19-prog](https://github.com/smgibson19-prog)
- **Issues:** [Open an issue](https://github.com/smgibson19-prog/Lux-creative-969/issues) for questions or feedback

---

*Made with 💙 by smgibson19-prog*
