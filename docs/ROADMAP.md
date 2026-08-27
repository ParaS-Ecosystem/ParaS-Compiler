
# ParaS Compiler Roadmap

**Repository:** <https://github.com/ParaS-Ecosystem/paras-compiler>  
**Project:** ParaS Compiler  
**Roadmap status:** Proposed community roadmap  
**Last updated:** July 2026

---

## 1. Vision

ParaS Compiler aims to provide an open, device-agnostic programming foundation for high performance computing, AI and scientific applications. The long-term goal is to allow developers to write modern C++/SYCL applications once and execute them efficiently across diverse CPU, GPU and future accelerator architectures.

The guiding principle is:

> **Code Once, Execute on All**

ParaS Compiler is positioned as the compiler foundation of the broader ParaS Ecosystem, enabling portable application development, backend portability, performance portability and community-driven innovation.

---

## 2. Current Public Repository Status

The current open-source repository provides the initial public base for ParaS Compiler. As available in the repository at the time of preparing this roadmap:

- ParaS Compiler is described as an implementation of the **SYCL 2020 specification**.
- The current release focuses on **CPU and Nvidia GPU execution support**.
- Build support is provided through **CMake**.
- The compiler is built using **Clang/LLVM v21.1.0** as the base compiler.
- The repository contains initial source, headers, examples, CMake files and GitHub workflow structure.
- The primary user-facing compiler driver is `parascc`.
- The project is licensed under **Apache-2.0**.

This roadmap builds from this current public baseline and proposes a staged path for making ParaS Compiler mature, usable and attractive to contributors.

---

## 3. Roadmap Principles

The roadmap follows five core principles:

1. **Open and community-first:** Make the project easy to build, test, use and contribute to.
2. **Standards-aligned:** Track SYCL 2020 compliance and future SYCL evolution in a transparent manner.
3. **Device agnostic:** Support multiple CPU, GPU and accelerator backends through a unified programming model.
4. **Performance portable:** Enable applications to achieve good performance across architectures without major code rewrites.
5. **Ecosystem integrated:** Work closely with ParaS libraries, tools, frameworks and applications.

---

## 4. Near-Term Roadmap: Open-Source Readiness

### 4.1 Repository Readiness

**Goal:** Make the repository easy for new users and contributors to understand.

Planned work:

- Add or refine `CONTRIBUTING.md`.
- Add `CODE_OF_CONDUCT.md`.
- Add `SECURITY.md` for responsible vulnerability reporting.
- Add issue templates for:
  - Bug reports
  - Feature requests
  - Backend support requests
  - Documentation improvements
- Add pull request template with checklist for build, test and documentation.
- Add `CHANGELOG.md` for release history.
- Add `MAINTAINERS.md` with maintainer areas and review ownership.

Expected outcome:

- Contributors can clearly understand how to participate.
- The project becomes ready for external community collaboration.

---

### 4.2 Build and Installation Improvements

**Goal:** Make ParaS Compiler easy to build on common Linux systems and HPC environments.

Planned work:

- Improve installation instructions for:
  - Ubuntu/Debian
  - RHEL/Rocky/AlmaLinux
  - HPC cluster environments
- Add dependency installation notes.
- Add CMake preset files for standard configurations.
- Add documented build modes:
  - Debug
  - Release
  - Developer build
- Add smoke-test instructions after installation.
- Add troubleshooting section for common build and runtime issues.
- Provide container-based build option for reproducibility.

Expected outcome:


- New developers can build ParaS Compiler with minimal manual intervention.
- HPC users can deploy ParaS Compiler in cluster environments more easily.

---

### 4.3 Continuous Integration and Quality Gates

**Goal:** Establish reliable CI for every pull request.

Planned work:

- Enable CI for build validation.
- Add formatting checks using `clang-format`.
- Add static analysis checks where feasible.
- Add basic unit tests and smoke tests.
- Add example compilation tests using `parascc`.
- Add CI status badges in `README.md`.
- Add release build validation workflow.

Expected outcome:

- Pull requests are validated automatically.
- Code quality and build stability improve over time.

---

## 5. Compiler and SYCL Roadmap

### 5.1 SYCL 2020 Core Feature Stabilization

**Goal:** Strengthen SYCL 2020 support for practical application development.

Planned work:

- Document supported SYCL 2020 features.
- Add a SYCL feature-compliance matrix.
- Stabilize support for:
  - Queues
  - Unified Shared Memory where applicable
  - Kernels and command groups
  - Device discovery
  - Runtime error handling
- Add regression tests for supported SYCL constructs.
- Add known limitations section.

Expected outcome:

- Users can clearly understand what is supported.
- Developers can track conformance progress transparently.

---

### 5.2 Runtime and Driver Improvements

**Goal:** Make `parascc` robust, predictable and developer friendly.

Planned work:

- Improve command-line help and diagnostics.
- Add verbose mode for debugging compilation flow.
- Improve error messages for unsupported flags and invalid device selections.
- Add documented environment variables.
- Add support for target/device selection flags.
- Add runtime logging controls.
- Improve integration with Clang/LLVM toolchain flags.

Expected outcome:

- Developers get clearer diagnostics.
- Application porting becomes easier.

---

### 5.3 Test Suite Expansion

**Goal:** Build confidence in correctness and portability.

Planned work:

- Add unit tests for runtime components.
- Add compiler driver tests.
- Add SYCL language feature tests.
- Add examples-based tests.
- Add numerical validation tests.
- Add performance sanity tests for key kernels.
- Organize tests by feature and backend.

Expected outcome:

- Feature additions and backend work can be validated reliably.
- Regressions can be detected early.

---

## 6. Backend Roadmap

### 6.1 CPU Backend Stabilization

**Goal:** Make CPU backend support stable and production usable.

Planned work:

- Improve CPU backend documentation.
- Validate on x86 platforms.
- Validate on ARM platforms.
- Add CPU backend conformance tests.
- Add OpenMP/runtime optimization studies.
- Add multi-core scaling examples.
- Add performance baselines for representative kernels.

Expected outcome:

- ParaS Compiler becomes a reliable SYCL programming path for CPU-based HPC systems.

---

### 6.2 AMD GPU Backend Enablement

**Goal:** Enable AMD GPU support to strengthen portability on modern HPC systems.

Planned work:

- Define HIP/ROCm backend integration options.
- Add build configuration for ROCm-based environments.
- Add AMD GPU device discovery.
- Add initial kernel compilation and execution support.
- Add examples for AMD GPU systems.
- Validate with representative HPC and AI kernels.

Expected outcome:

- ParaS Compiler supports a broader accelerator ecosystem and avoids dependence on a single GPU vendor.

---

### 6.3 RISC-V CPU Exploration

**Goal:** Prepare ParaS Compiler for emerging RISC-V HPC platforms.

Planned work:

- Validate build support on RISC-V Linux platforms.
- Add RISC-V target documentation.
- Add simple SYCL examples for RISC-V CPU execution.
- Study vectorization opportunities through LLVM and RISC-V vector extensions.
- Add portability tests for core runtime features.

Expected outcome:

- ParaS Compiler becomes ready for early RISC-V HPC experimentation.

---

### 6.4 Future Accelerator Support

**Goal:** Keep the architecture open for NPUs, AI accelerators and future XPUs.

Planned work:

- Define backend plugin/interface expectations.
- Document requirements for adding a new backend.
- Explore runtime abstraction for future accelerators.
- Add design notes for accelerator-specific memory and execution models.
- Encourage external hardware teams to contribute backend prototypes.
- Define runtime abstraction for Gaudi3 accelerator support.
- Define runtime abstraction for Sambhanova accelerator support.

Expected outcome:

- ParaS Compiler remains extensible for future heterogeneous systems.

---

## 7. Documentation Roadmap

### 7.1 User Documentation

Planned work:

- Quick start guide.
- Installation guide.
- First SYCL program using ParaS Compiler.
- Compiler flags reference.
- Runtime environment reference.
- Backend selection guide.
- FAQ and troubleshooting guide.


### 7.2 Examples and Tutorials

Planned work:

- Stencil computation.
- Memory transfer examples.
- Multi-device examples.
- HPC mini-app examples.

Expected outcome:

- Application developers, system administrators and contributors can all use the project with lower entry barriers.

---

## 8. Performance Roadmap

**Goal:** Make performance measurement transparent and reproducible.

Planned work:

- Add benchmark directory.
- Add baseline kernels:
  - Vector add
  - Matrix multiplication
  - Reduction
  - Stencil
  - STREAM-like memory bandwidth kernel
- Add scripts for reproducible performance runs.
- Add CPU performance baselines.
- Add GPU performance baselines as backends mature.
- Add comparison reports across architectures.
- Add tuning notes for common platforms.

Expected outcome:

- Performance portability can be measured, improved and demonstrated publicly.

---

## 9. Ecosystem Integration Roadmap

ParaS Compiler should evolve together with the broader ParaS Ecosystem.

Planned integrations:

- ParaS Math and numerical libraries.
- ParaS Collective Communication Library.
- ParaS framework backends.
- ParaS applications repository.
- ParaS profiling and debugging tools.
- IDE/developer tooling integration.
- Container and package deployment workflows.

Expected outcome:

- ParaS Compiler becomes the foundation for a complete indigenous, open, device-agnostic HPC-AI programming ecosystem.

---

## 10. Packaging and Release Roadmap

### 10.1 Initial Open-Source Releases

Planned work:

- Publish versioned source releases.
- Define semantic versioning policy.
- Provide release notes.
- Provide checksums for release artifacts.
- Provide reproducible build instructions.

### 10.2 Binary and Container Distribution

Planned work:

- Provide container images for quick evaluation.
- Provide package scripts where feasible.
- Add Spack package recipe exploration.
- Add modulefile examples for HPC clusters.

Expected outcome:

- Users can install and evaluate ParaS Compiler without building everything manually.

---

## 11. Suggested Milestones

### Milestone 0: Open-Source Foundation

Focus:

- Repository governance files.
- Build documentation.
- CI foundation.
- Basic examples.
- Roadmap publication.

Indicative status target:

- Ready for first external contributors.

---

### Milestone 1: CPU-Ready Developer Release

Focus:

- Stable CPU execution path.
- SYCL feature matrix.
- Improved `parascc` diagnostics.
- Expanded examples and tests.
- Container-based evaluation flow.

Indicative status target:

- Usable by application developers for CPU-based SYCL experimentation and early porting.

---

### Milestone 2: Multi-Backend Preview

Focus:

- NVIDIA GPU backend preview.
- AMD GPU backend design and initial prototype.
- Backend selection interface.
- Multi-backend tests.
- Initial performance baselines.

Indicative status target:

- Demonstrates the path from CPU-only to heterogeneous execution.

---

### Milestone 3: HPC-AI Ecosystem Integration

Focus:

- Integration with ParaS Math and related libraries.
- Integration with selected HPC mini-apps.
- Integration with selected AI framework/backend experiments.
- Profiling/debugging workflow documentation.

Indicative status target:

- Demonstrates real scientific and AI workloads through ParaS Compiler.

---

### Milestone 4: Production-Oriented Release

Focus:

- Hardened CI and test coverage.
- Versioned release artifacts.
- Installation packages or containers.
- Clear backend support matrix.
- Performance reports.
- Mature contributor workflow.

Indicative status target:

- Ready for wider adoption across HPC centres, academic users and application teams.

---

## 12. Contribution Areas

The project welcomes contributions in the following areas:

- SYCL feature implementation such as:
      - Kernel Bundle API
      - Reduction API
      - Images
      - Samplers
- CPU backend optimization.
- GPU backend development.
- RISC-V platform validation.
- Testing and conformance.
- Documentation and tutorials.
- Packaging and containers.
- Benchmarks and performance analysis.
- HPC and AI example applications.

Suggested labels for GitHub issues:

- `good first issue`
- `help wanted`
- `documentation`
- `build`
- `testing`
- `runtime`
- `compiler-driver`
- `cpu-backend`
- `gpu-backend`
- `riscv`
- `performance`
- `ecosystem-integration`
- `roadmap`

---

## 13. Definition of Done for Roadmap Items

A roadmap item should be considered complete only when it includes:

- Working implementation or documentation.
- Tests or validation steps where applicable.
- Updated user/developer documentation.
- CI coverage where practical.
- Known limitations documented.
- Review and approval by maintainers.

---

## 14. Notes to Contributors

This roadmap is intentionally open and may evolve based on community feedback, hardware availability, standards evolution and ecosystem requirements.

Contributors are encouraged to:

- Start with documentation, examples or tests.
- Open issues before large design changes.
- Share platform-specific validation results.
- Contribute backend enablement work in incremental steps.
- Help make ParaS Compiler a practical, open and device-agnostic programming foundation for HPC and AI.

---

## 15. Summary

ParaS Compiler begins its open-source journey with a focused SYCL 2020 CPU execution base. The proposed roadmap moves the project toward a mature, multi-backend, performance-portable compiler infrastructure that can serve Indian and global HPC-AI application developers.

The immediate priority is to strengthen repository readiness, build reliability, documentation, testing and CPU backend stability. The next major step is to expand toward GPU and emerging architecture support while growing a strong open-source contributor community.

