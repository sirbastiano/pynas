# draw_model_graph.py
import argparse
import os
from typing import List, Tuple

import torch
from graphviz import Digraph
from graphviz.backend.execute import ExecutableNotFound

try:
    from torchviz import make_dot
    _HAS_TORCHVIZ = True
except Exception:
    _HAS_TORCHVIZ = False


def parse_shape(s: str) -> Tuple[int, ...]:
    """Parse a comma-separated shape string like '1,3,256,256' into a tuple of ints."""
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def build_module_hierarchy_graph(model: torch.jit.ScriptModule, outpath: str) -> None:
    """Render the nn.Module hierarchy (tree) using Graphviz.

    Args:
        model: Loaded TorchScript model.
        outpath: Output file path without extension (Graphviz will add .pdf/.png).
    """
    dot = Digraph(
        "module_hierarchy",
        format="pdf",
        node_attr={"shape": "box", "fontname": "Helvetica"},
        graph_attr={"rankdir": "LR"}
    )

    # Collect all modules with their qualified names
    name_to_module = dict(model.named_modules())
    # Root in named_modules() appears as '' (empty string)
    # We create a readable root label.
    def node_label(name: str, module: torch.nn.Module) -> str:
        cls = module.original_name if hasattr(module, "original_name") else module.__class__.__name__
        return f"{name or 'model'}\\n({cls})"

    # Add nodes
    for name, module in name_to_module.items():
        dot.node(name or "model", label=node_label(name, module))

    # Add edges parent -> child
    for name in name_to_module.keys():
        if name == "":
            continue
        parent = name.rsplit(".", 1)[0]
        parent = parent if parent else "model"
        dot.edge(parent, name)

    try:
        dot.render(outpath, cleanup=True)
    except ExecutableNotFound as e:
        # Fallback: write the DOT source so the user can render it manually
        dot_path = f"{outpath}.dot"
        try:
            with open(dot_path, "w") as f:
                f.write(dot.source)
            print(
                f"Graphviz 'dot' executable not found on PATH. Wrote DOT source to {dot_path}."
            )
            print(
                "Install Graphviz (provides `dot`) to render PDFs, e.g.:\n"
                "  Debian/Ubuntu: sudo apt-get install graphviz\n"
                "  Fedora: sudo dnf install graphviz\n"
                "  macOS (Homebrew): brew install graphviz"
            )
        except Exception:
            # If writing the file fails, re-raise original exception for visibility
            raise


@torch.no_grad()
def build_computation_graph(
    model: torch.jit.ScriptModule,
    input_shape: Tuple[int, ...],
    dtype: str,
    device: str,
    outpath: str,
) -> None:
    """Run a dummy forward pass and draw the autograd computation graph (ops graph) with torchviz.

    Notes:
        - Requires torchviz. If not available, this function raises a RuntimeError.
        - The model must accept a single tensor of shape `input_shape` as first argument.
        - If your model requires multiple inputs, adapt the `dummy_inputs` construction accordingly.
    """
    if not _HAS_TORCHVIZ:
        raise RuntimeError(
            "torchviz is not installed. Install with `pip install torchviz` to enable computation graph."
        )

    torch_dtype = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }.get(dtype.lower(), torch.float32)

    model = model.to(device)
    model.eval()

    x = torch.zeros(input_shape, dtype=torch_dtype, device=device, requires_grad=False)
    # If your model expects more inputs, modify here, e.g.: outputs = model(x, aux)
    outputs = model(x)

    # torchviz wants a tensor (or sequence). If ScriptModule returns dict, pick a tensor value.
    if isinstance(outputs, dict):
        # Heuristic: pick the first tensor value
        for v in outputs.values():
            if torch.is_tensor(v):
                outputs = v
                break

    if not torch.is_tensor(outputs):
        raise RuntimeError(
            "Model output is not a Tensor. Adapt the code to select a tensor output for visualization."
        )

    dot = make_dot(outputs, params=dict(list(model.named_parameters())))
    dot.format = "pdf"
    dot.directory = os.path.dirname(os.path.abspath(outpath)) or "."
    base = os.path.splitext(os.path.basename(outpath))[0]
    try:
        dot.render(base, cleanup=True)
    except ExecutableNotFound:
        # Write DOT file in the same directory as outpath for manual rendering
        dot_path = os.path.join(dot.directory or ".", f"{base}.dot")
        try:
            with open(dot_path, "w") as f:
                f.write(dot.source)
            print(f"Graphviz 'dot' executable not found. Wrote DOT source to {dot_path}.")
            print(
                "Install Graphviz to render the computation graph to PDF, e.g.:\n"
                "  Debian/Ubuntu: sudo apt-get install graphviz\n"
                "  Fedora: sudo dnf install graphviz\n"
                "  macOS (Homebrew): brew install graphviz"
            )
        except Exception:
            raise


def export_torchscript_ir(model: torch.jit.ScriptModule, out_txt: str) -> None:
    """Dump the TorchScript IR (inlined graph) to a text file for inspection."""
    try:
        ir = str(model.inlined_graph)
    except Exception:
        ir = str(model.graph)
    with open(out_txt, "w") as f:
        f.write(ir)


def main():
    parser = argparse.ArgumentParser(
        description="Draw graphs for a TorchScript model: module hierarchy and (optionally) computation graph."
    )
    parser.add_argument("--model", required=True, help="Path to .pt/.pth TorchScript file (ScriptModule archive).")
    parser.add_argument("--hier-out", default="model_hierarchy.pdf", help="Output PDF for module hierarchy.")
    parser.add_argument("--comp-out", default="computation_graph.pdf", help="Output PDF for computation graph.")
    parser.add_argument("--ir-out", default="model_ir.txt", help="Output text file with TorchScript IR.")
    parser.add_argument("--input-shape", default=None, help="Comma-separated shape, e.g. 1,3,256,256")
    parser.add_argument("--dtype", default="float32", help="Input dtype (float32|float16|bfloat16).")
    parser.add_argument("--device", default="cpu", help="Device for dummy run (cpu|cuda).")
    args = parser.parse_args()

    # Load TorchScript archive (warnings about zip/ScriptModule are OK)
    model = torch.jit.load(args.model, map_location="cpu")

    # 1) Module hierarchy
    build_module_hierarchy_graph(model, os.path.splitext(args.hier_out)[0])

    # 2) TorchScript IR dump
    export_torchscript_ir(model, args.ir_out)

    # 3) Optional computation graph (requires a *valid* input shape and torchviz)
    if args.input_shape is not None:
        shape = parse_shape(args.input_shape)
        build_computation_graph(
            model=model,
            input_shape=shape,
            dtype=args.dtype,
            device=args.device,
            outpath=args.comp_out,
        )

    print(f"Saved: {args.hier_out}")
    print(f"Saved: {args.ir_out}")
    if args.input_shape is not None:
        print(f"Saved: {args.comp_out}")


if __name__ == "__main__":
    main()