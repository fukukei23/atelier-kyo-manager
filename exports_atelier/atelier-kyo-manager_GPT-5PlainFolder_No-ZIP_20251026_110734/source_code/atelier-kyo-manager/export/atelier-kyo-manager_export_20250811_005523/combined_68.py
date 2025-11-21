
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\onnx_model.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import itertools
import logging
import os
import sys
from collections import deque
from pathlib import Path

from float16 import convert_float_to_float16
from onnx import (
    AttributeProto,
    GraphProto,
    ModelProto,
    NodeProto,
    TensorProto,
    ValueInfoProto,
    helper,
    numpy_helper,
    save_model,
)
from onnx.external_data_helper import load_external_data_for_tensor, uses_external_data
from shape_infer_helper import SymbolicShapeInferenceHelper

logger = logging.getLogger(__name__)


class OnnxModel:
    def __init__(self, model):
        self.initialize(model)

    def initialize(self, model):
        self.model: ModelProto = model
        self._node_name_suffix: dict[str, int] = {}  # key is node name prefix, value is the last suffix generated
        self.shape_infer_helper: SymbolicShapeInferenceHelper = None
        self.enable_shape_infer: bool = True
        self.all_graphs: list[GraphProto] | None = None

        # Cache of shape and data type from onnx graph to speed up optimization.
        # Be careful that fusion shall not reuse node output name for different shape/type (in adding/removing nodes)
        # Note that these do not cache the symbolic shape inference result.
        self._dtype_dict: dict[str, int] | None = None
        self._shape_dict: dict[str, list] | None = None

    def disable_shape_inference(self):
        self.enable_shape_infer = False

    def infer_runtime_shape(self, dynamic_axis_mapping={}, update=False):  # noqa: B006
        if self.enable_shape_infer:
            if self.shape_infer_helper is None or update:
                self.shape_infer_helper = SymbolicShapeInferenceHelper(self.model)

            try:
                if self.shape_infer_helper.infer(dynamic_axis_mapping):
                    return self.shape_infer_helper
            except Exception:
                self.enable_shape_infer = False  # disable shape inference to suppress same error message.
                print("failed in shape inference", sys.exc_info()[0])

        return None

    def input_name_to_nodes(self, exclude_subgraphs=False):
        input_name_to_nodes = {}
        nodes_to_search = self.nodes() if not exclude_subgraphs else self.model.graph.node
        for node in nodes_to_search:
            for input_name in node.input:
                if input_name:  # could be empty when it is optional
                    if input_name not in input_name_to_nodes:
                        input_name_to_nodes[input_name] = [node]
                    else:
                        input_name_to_nodes[input_name].append(node)
        return input_name_to_nodes

    def output_name_to_node(self, exclude_subgraphs=False):
        output_name_to_node = {}
        nodes_to_search = self.nodes() if not exclude_subgraphs else self.model.graph.node
        for node in nodes_to_search:
            for output_name in node.output:
                if output_name:  # could be empty when it is optional
                    output_name_to_node[output_name] = node
        return output_name_to_node

    def functions(self):
        all_functions = [list(self.model.functions)]
        return all_functions

    def nodes(self):
        all_nodes = []
        for graph in self.graphs():
            for node in graph.node:
                all_nodes.append(node)  # noqa: PERF402
        return all_nodes

    def graph(self):
        return self.model.graph

    def graphs(self):
        if self.all_graphs is not None:
            return self.all_graphs
        self.all_graphs = []
        graph_queue = [self.model.graph]
        while graph_queue:
            graph = graph_queue.pop(0)
            self.all_graphs.append(graph)
            for node in graph.node:
                for attr in node.attribute:
                    if attr.type == AttributeProto.AttributeType.GRAPH:
                        assert isinstance(attr.g, GraphProto)
                        graph_queue.append(attr.g)
                    if attr.type == AttributeProto.AttributeType.GRAPHS:
                        for g in attr.graphs:
                            assert isinstance(g, GraphProto)
                            graph_queue.append(g)
        return self.all_graphs

    def get_graphs_input_names(self):
        input_names = []
        for graph in self.graphs():
            for input in graph.input:
                input_names.append(input.name)
        return input_names

    def get_graphs_output_names(self):
        output_names = []
        for graph in self.graphs():
            for output in graph.output:
                output_names.append(output.name)
        return output_names

    def get_graph_by_node(self, node):
        for graph in self.graphs():
            if node in graph.node:
                return graph
        return None

    def get_graph_by_name(self, graph_name):
        for graph in self.graphs():
            if graph_name == graph.name:
                return graph
        return None

    def get_topological_insert_id(self, graph, outputs):
        for idx, node in enumerate(graph.node):
            for input in node.input:
                if input in outputs:
                    return idx
        return len(graph.node)

    def remove_node(self, node):
        for graph in self.graphs():
            if node in graph.node:
                graph.node.remove(node)
                return
        logger.warning("Failed to remove node %s", node)  # It might be a bug to hit this line.

    def remove_nodes(self, nodes_to_remove):
        for node in nodes_to_remove:
            self.remove_node(node)

    def add_node(self, node, graph_name=None):
        if graph_name is None or graph_name == self.model.graph.name:
            self.model.graph.node.extend([node])
        else:
            graph = self.get_graph_by_name(graph_name)
            insert_idx = self.get_topological_insert_id(graph, node.output)
            graph.node.insert(insert_idx, node)

    def add_nodes(self, nodes_to_add, node_name_to_graph_name=None):
        if node_name_to_graph_name is None:
            self.model.graph.node.extend(nodes_to_add)
        else:
            for node in nodes_to_add:
                graph_name = node_name_to_graph_name[node.name]
                self.add_node(node, graph_name)

    def add_initializer(self, tensor, graph_name=None):
        if graph_name is None or graph_name == self.model.graph.name:
            self.model.graph.initializer.extend([tensor])
        else:
            graph = self.get_graph_by_name(graph_name)
            graph.initializer.extend([tensor])

    def add_input(self, input, graph_name=None):
        if graph_name is None or graph_name == self.model.graph.name:
            self.model.graph.input.extend([input])
        else:
            graph = self.get_graph_by_name(graph_name)
            graph.input.extend([input])

    @staticmethod
    def replace_node_input(node, old_input_name, new_input_name):
        assert isinstance(old_input_name, str) and isinstance(new_input_name, str)
        for j in range(len(node.input)):
            if node.input[j] == old_input_name:
                node.input[j] = new_input_name

    def replace_input_of_all_nodes(self, old_input_name, new_input_name):
        for node in self.nodes():
            OnnxModel.replace_node_input(node, old_input_name, new_input_name)

    @staticmethod
    def replace_node_output(node, old_output_name, new_output_name):
        assert isinstance(old_output_name, str) and isinstance(new_output_name, str)
        for j in range(len(node.output)):
            if node.output[j] == old_output_name:
                node.output[j] = new_output_name

    def replace_output_of_all_nodes(self, old_output_name, new_output_name):
        # This function shall be used carefully. For example:
        #       Add --[old_name]--> Cast ---> [new_name]
        #        |
        #        +----[old_name]--> Transpose -->
        # If we want to remove the Cast node: replace output of Add to new_name is not enough;
        # The input of Transpose shall also be updated to new_name.
        for node in self.model.graph.node:
            OnnxModel.replace_node_output(node, old_output_name, new_output_name)

    def get_initializer(self, name):
        for graph in self.graphs():
            for tensor in graph.initializer:
                if tensor.name == name:
                    return tensor
        return None

    def get_nodes_by_op_type(self, op_type):
        nodes = []
        for node in self.nodes():
            if node.op_type == op_type:
                nodes.append(node)
        return nodes

    def get_children(self, node, input_name_to_nodes=None, output_index=None):
        if input_name_to_nodes is None:
            input_name_to_nodes = self.input_name_to_nodes()

        children = []
        if output_index is not None:
            if output_index < len(node.output):
                output = node.output[output_index]
                if output in input_name_to_nodes:
                    children = list(input_name_to_nodes[output])
        else:
            for output in node.output:
                if output in input_name_to_nodes:
                    children.extend(input_name_to_nodes[output])

        return children

    def get_parents(self, node, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        parents = []
        for input in node.input:
            if input in output_name_to_node:
                parents.append(output_name_to_node[input])
        return parents

    def get_parent(self, node, i, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        if len(node.input) <= i:
            return None

        input = node.input[i]
        if input not in output_name_to_node:
            return None

        return output_name_to_node[input]

    def match_first_parent(self, node, parent_op_type, output_name_to_node, exclude=[]):  # noqa: B006
        """
        Find parent node based on constraints on op_type.

        Args:
            node (str): current node name.
            parent_op_type (str): constraint of parent node op_type.
            output_name_to_node (dict): dictionary with output name as key, and node as value.
            exclude (list): list of nodes that are excluded (not allowed to match as parent).

        Returns:
            parent: The matched parent node. None if not found.
            index: The input index of matched parent node. None if not found.
        """
        for i, input in enumerate(node.input):
            if input in output_name_to_node:
                parent = output_name_to_node[input]
                if parent.op_type == parent_op_type and parent not in exclude:
                    return parent, i
                else:
                    logger.debug(f"To find first {parent_op_type}, current {parent.op_type}")
        return None, None

    def match_parent(
        self,
        node,
        parent_op_type,
        input_index=None,
        output_name_to_node=None,
        exclude=[],  # noqa: B006
        return_indice=None,
    ):
        """
        Find parent node based on constraints on op_type and index.
        When input_index is None, we will find the first parent node based on constraints,
        and return_indice will be appended the corresponding input index.

        Args:
            node (str): current node name.
            parent_op_type (str): constraint of parent node op_type.
            input_index (int or None): only check the parent given input index of current node.
            output_name_to_node (dict): dictionary with output name as key, and node as value.
            exclude (list): list of nodes that are excluded (not allowed to match as parent).
            return_indice (list): a list to append the input index when input_index is None.

        Returns:
            parent: The matched parent node.
        """
        assert node is not None
        assert input_index is None or input_index >= 0

        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        if input_index is None:
            parent, index = self.match_first_parent(node, parent_op_type, output_name_to_node, exclude)
            if return_indice is not None:
                return_indice.append(index)
            return parent

        if input_index >= len(node.input):
            logger.debug(f"input_index {input_index} >= node inputs {len(node.input)}")
            return None

        parent = self.get_parent(node, input_index, output_name_to_node)
        if parent is not None and parent.op_type == parent_op_type and parent not in exclude:
            return parent

        if parent is not None:
            logger.debug(f"Expect {parent_op_type}, Got {parent.op_type}")

        return None

    def match_parent_paths(self, node, paths, output_name_to_node):
        for i, path in enumerate(paths):
            assert isinstance(path, (list, tuple))
            return_indice = []
            matched = self.match_parent_path(node, path[0], path[1], output_name_to_node, return_indice)
            if matched:
                return i, matched, return_indice
        return -1, None, None

    def match_parent_paths_all(self, node, paths, output_name_to_node):
        match_i, matches, return_indices = [], [], []
        for i, path in enumerate(paths):
            assert isinstance(path, (list, tuple))
            return_indice = []
            matched = self.match_parent_path(node, path[0], path[1], output_name_to_node, return_indice)
            if matched:
                match_i.append(i)
                matches.append(matched)
                return_indices.append(return_indice)
        return match_i, matches, return_indices

    def match_parent_path(
        self,
        node,
        parent_op_types,
        parent_input_index=None,
        output_name_to_node=None,
        return_indice=None,
    ):
        """
        Find a sequence of input edges based on constraints on parent op_type and index.
        When input_index is None, we will find the first parent node based on constraints,
        and return_indice will be appended the corresponding input index.

        Args:
            node (str): current node name.
            parent_op_types (str): constraint of parent node op_type of each input edge.
            parent_input_index (list): constraint of input index of each input edge. None means no constraint.
            output_name_to_node (dict): dictionary with output name as key, and node as value.
            return_indice (list): a list to append the input index
                                  When there is no constraint on input index of an edge.

        Returns:
            parents: a list of matched parent node.
        """
        if parent_input_index is not None:
            assert len(parent_input_index) == len(parent_op_types)

        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        current_node = node
        matched_parents = []
        for i, op_type in enumerate(parent_op_types):
            matched_parent = self.match_parent(
                current_node,
                op_type,
                parent_input_index[i] if parent_input_index is not None else None,
                output_name_to_node,
                exclude=[],
                return_indice=return_indice,
            )
            if matched_parent is None:
                if parent_input_index is not None:
                    logger.debug(
                        f"Failed to match index={i} parent_input_index={parent_input_index[i]} op_type={op_type}",
                        stack_info=True,
                    )
                else:
                    logger.debug(f"Failed to match index={i} op_type={op_type}", stack_info=True)
                return None

            matched_parents.append(matched_parent)
            current_node = matched_parent

        return matched_parents

    def find_first_child_by_type(self, node, child_type, input_name_to_nodes=None, recursive=True):
        children = self.get_children(node, input_name_to_nodes)
        dq = deque(children)
        while len(dq) > 0:
            current_node = dq.pop()
            if current_node.op_type == child_type:
                return current_node

            if recursive:
                children = self.get_children(current_node, input_name_to_nodes)
                for child in children:
                    dq.appendleft(child)

        return None

    def match_child_path(
        self,
        node,
        child_op_types,
        edges: list[tuple[int, int]] | None = None,
        input_name_to_nodes=None,
        exclude=[],  # noqa: B006
    ):
        """
        Find a sequence of input edges based on constraints on parent op_type and index.
        Note that we use greedy approach and only consider the first matched child, so it has chance to miss matching.

        Args:
            node (str): current node name.
            child_op_types (str): constraint of child node op_type of each input edge.
            edges (list): each edge is represented by two integers: output index of parent node, input index of child node.
                         None means no constraint.
            exclude(list): list of nodes that are excluded (not allowed to match as child).

        Returns:
            children: a list of matched children node.
        """
        if edges is not None:
            assert len(edges) == len(child_op_types)
            for edge in edges:
                assert (
                    isinstance(edge, tuple) and len(edge) == 2 and isinstance(edge[0], int) and isinstance(edge[1], int)
                )

        if input_name_to_nodes is None:
            input_name_to_nodes = self.input_name_to_nodes()

        current_node = node
        matched_children = []
        for i, op_type in enumerate(child_op_types):
            matched_child = None

            if edges is None:
                children_nodes = self.get_children(current_node, input_name_to_nodes=input_name_to_nodes)
            else:
                children_nodes = self.get_children(
                    current_node, input_name_to_nodes=input_name_to_nodes, output_index=edges[i][0]
                )

            for child in children_nodes:
                if child.op_type == op_type and child not in exclude:
                    if edges is not None and child.input[edges[i][1]] != current_node.output[edges[i][0]]:
                        continue

                    # Here we use greedy approach and only consider the first matched child.
                    # TODO: match recursively if we encounter cases that the correct child is not the first matched.
                    matched_child = child
                    break

            if matched_child is None:
                logger.debug(f"Failed to match child {i} op_type={op_type}", stack_info=True)
                return None

            matched_children.append(matched_child)
            current_node = matched_child

        return matched_children

    def find_first_parent_by_type(self, node, parent_type, output_name_to_node=None, recursive=True):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        parents = self.get_parents(node, output_name_to_node)
        dq = deque(parents)
        while len(dq) > 0:
            current_node = dq.pop()
            if current_node.op_type == parent_type:
                return current_node

            if recursive:
                parents = self.get_parents(current_node, output_name_to_node)
                for parent in parents:
                    dq.appendleft(parent)

        return None

    def get_constant_value(self, output_name):
        for node in self.get_nodes_by_op_type("Constant"):
            if node.output[0] == output_name:
                for att in node.attribute:
                    if att.name == "value":
                        return numpy_helper.to_array(att.t)

        # Fall back to intializer since constant folding might have been applied.
        initializer = self.get_initializer(output_name)
        if initializer is not None:
            return numpy_helper.to_array(initializer)

        return None

    def get_constant_input(self, node):
        for i, input in enumerate(node.input):
            value = self.get_constant_value(input)
            if value is not None:
                return i, value

        return None, None

    def find_constant_input(self, node, expected_value, delta=0.000001):
        i, value = self.get_constant_input(node)
        if value is not None and value.size == 1 and abs(value - expected_value) < delta:
            return i

        return -1

    def is_constant_with_specified_dimension(self, output_name, dimensions, description):
        value = self.get_constant_value(output_name)
        if value is None:
            logger.debug(f"{description} {output_name} is not initializer.")
            return False

        if len(value.shape) != dimensions:
            logger.debug(f"{description} {output_name} shall have {dimensions} dimensions. Got shape {value.shape}")
            return False

        return True

    def has_constant_input(self, node, expected_value, delta=0.000001):
        return self.find_constant_input(node, expected_value, delta) >= 0

    def get_children_subgraph_nodes(self, root_node, stop_nodes, input_name_to_nodes=None):
        if input_name_to_nodes is None:
            input_name_to_nodes = self.input_name_to_nodes()

        children = input_name_to_nodes[root_node.output[0]]

        unique_nodes = []

        dq = deque(children)
        while len(dq) > 0:
            current_node = dq.pop()
            if current_node in stop_nodes:
                continue

            if current_node not in unique_nodes:
                unique_nodes.append(current_node)

                for output in current_node.output:
                    if output in input_name_to_nodes:
                        children = input_name_to_nodes[output]
                        for child in children:
                            dq.appendleft(child)

        return unique_nodes

    def tensor_shape_to_list(self, tensor_type):
        """Convert tensor shape to list"""
        shape_list = []
        for d in tensor_type.shape.dim:
            if d.HasField("dim_value"):
                shape_list.append(d.dim_value)  # known dimension
            elif d.HasField("dim_param"):
                shape_list.append(d.dim_param)  # unknown dimension with symbolic name
            else:
                shape_list.append("?")  # shall not happen
        return shape_list

    def get_dtype(self, name: str, symbolic_shape_helper: SymbolicShapeInferenceHelper | None = None):
        """Try get data type given a name (could be initializer, input or output of graph or node)."""

        if self._dtype_dict is None:
            self._dtype_dict = {}
            for value_info in itertools.chain(
                self.model.graph.value_info,
                self.model.graph.input,
                self.model.graph.output,
            ):
                self._dtype_dict[value_info.name] = value_info.type.tensor_type.elem_type

            for initializer in self.model.graph.initializer:
                if initializer.name not in self._dtype_dict:
                    self._dtype_dict[initializer.name] = initializer.data_type

        if name in self._dtype_dict:
            return self._dtype_dict[name]

        if symbolic_shape_helper is not None and name in symbolic_shape_helper.known_vi_:
            value_info = symbolic_shape_helper.known_vi_[name]
            return value_info.type.tensor_type.elem_type

        return None

    def get_shape(self, name: str, symbolic_shape_helper: SymbolicShapeInferenceHelper | None = None):
        """Try get shape given a name (could be initializer, input or output of graph or node)."""

        if self._shape_dict is None:
            self._shape_dict = {}
            for value_info in itertools.chain(
                self.model.graph.value_info,
                self.model.graph.input,
                self.model.graph.output,
            ):
                if value_info.type.tensor_type.HasField("shape"):
                    shape = []
                    for dim in value_info.type.tensor_type.shape.dim:
                        if dim.dim_param:
                            shape.append(dim.dim_param)
                        else:
                            shape.append(dim.dim_value)
                    self._shape_dict[value_info.name] = shape

            for initializer in self.model.graph.initializer:
                if initializer.name not in self._shape_dict:
                    self._shape_dict[initializer.name] = initializer.dims

        if name in self._shape_dict:
            return self._shape_dict[name]

        if symbolic_shape_helper is not None and name in symbolic_shape_helper.known_vi_:
            value_info = symbolic_shape_helper.known_vi_[name]
            return value_info.type.tensor_type.elem_type

        return None

    @staticmethod
    def get_node_attribute(node: NodeProto, attribute_name: str):
        for attr in node.attribute:
            if attr.name == attribute_name:
                value = helper.get_attribute_value(attr)
                return value
        return None

    def remove_cascaded_cast_nodes(self):
        """Remove Cast node that are followed by another Cast node like  --> Cast --> Cast -->
        Note that this shall be used carefully since it might introduce semantic change.
        For example, float -> int -> float could get different value than the original float value.
        So, it is recommended to used only in post-processing of mixed precision conversion.
        """
        output_name_to_node = self.output_name_to_node()
        removed_count = 0
        for node in self.nodes():
            if node.op_type == "Cast":
                parent = self.get_parent(node, 0, output_name_to_node=output_name_to_node)
                if parent and parent.op_type == "Cast":
                    node.input[0] = parent.input[0]
                    removed_count += 1

        if removed_count > 0:
            logger.info("Removed %d cascaded Cast nodes", removed_count)
            self.prune_graph()

    def remove_useless_cast_nodes(self):
        """Remove cast nodes that are not needed: input and output has same data type."""
        shape_infer = self.infer_runtime_shape(update=True)
        if self.enable_shape_infer and shape_infer is None:
            logger.warning("shape inference failed which might impact useless cast node detection.")

        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == "Cast":
                input_dtype = self.get_dtype(node.input[0], shape_infer)
                output_dtype = self.get_dtype(node.output[0], shape_infer)
                if input_dtype and input_dtype == output_dtype:
                    nodes_to_remove.append(node)

        if nodes_to_remove:
            graph_input_names = set(self.get_graphs_input_names())
            graph_output_names = set(self.get_graphs_output_names())
            for node in nodes_to_remove:
                if bool(set(node.output) & graph_output_names):
                    if (not bool(set(node.input) & graph_input_names)) and len(
                        self.input_name_to_nodes()[node.input[0]]
                    ) == 1:
                        self.replace_output_of_all_nodes(node.input[0], node.output[0])
                    else:
                        continue
                else:
                    self.replace_input_of_all_nodes(node.output[0], node.input[0])
                self.remove_node(node)

            logger.info(
                "Removed %d Cast nodes with output type same as input",
                len(nodes_to_remove),
            )

    def convert_model_float32_to_float16(self, cast_input_output=True):
        logger.warning(
            "The function convert_model_float32_to_float16 is deprecated. Use convert_float_to_float16 instead!"
        )
        self.convert_float_to_float16(use_symbolic_shape_infer=True, keep_io_types=cast_input_output)

    def convert_float_to_float16(self, use_symbolic_shape_infer=True, **kwargs):
        """Convert a model to half (default) or mixed precision.
           To use mixed precision, user need specify which graph inputs, outputs, operator type
           or list of nodes shall keep in float32.

           Note that the conversion might not proceed without type information for the whole graph.

           By default, we use symbolic shape inference to get type information. The benefit of symbolic shape inference
           is that it could handle fused operators in com.microsoft domain. Those operators cannot be handled in onnx shape
           inference so symbolic shape inference is recommended for optimized model.

           When symbolic shape inference is used (even if it failed), ONNX shape inference will be disabled.

           Note that onnx shape inference will fail for model larger than 2GB. For large model, you have to enable
           symbolic shape inference. If your model is not optimized, you can also use model path to call
           convert_float_to_float16 in float16.py (see https://github.com/microsoft/onnxruntime/pull/15067) to
           avoid the 2GB limit.

        Args:
            use_symbolic_shape_infer (bool, optional): use symbolic shape inference instead of onnx shape inference.
                                                       Defaults to True.
            keep_io_types (Union[bool, List[str]], optional): boolean or a list of float32 input/output names.
                                                              If True, model inputs/outputs should be left as float32.
                                                              Defaults to True.
            op_block_list (List[str], optional): List of operator types to leave as float32.
                                                 Defaults to None, which will use `float16.DEFAULT_OP_BLOCK_LIST`.
            node_block_list (List[str], optional): List of node names to leave as float32. Defaults to None.
            force_fp16_initializers(bool): force converting all float initializers to float16.
                                           Default to false.
            min_positive_val (float, optional): minimal positive value. Defaults to 1e-7.
            max_finite_val (float, optional): maximal finite value. Defaults to 1e4.
            force_fp16_inputs(Dict[str, List[int]]): Force the conversion of the inputs of some operators to float16, even if
                                                     this script's preference it to keep them in float32.
        """
        if "keep_io_types" not in kwargs:
            kwargs["keep_io_types"] = True

        model = self.model
        if use_symbolic_shape_infer:
            # Use symbolic shape inference since custom operators (like Gelu, SkipLayerNormalization etc)
            # are not recognized by onnx shape inference.
            shape_infer_helper = SymbolicShapeInferenceHelper(model)
            try:
                model_with_shape = shape_infer_helper.infer_shapes(model, auto_merge=True, guess_output_rank=False)

                # auto_merge might cause issue (see https://github.com/microsoft/onnxruntime/issues/15521)
                # we only merge tensor data type but not shape information back to the original onnx model.
                # Note that float16 conversion need data type but not shape information.
                if model_with_shape is not None:
                    name_vi = {}
                    for vi in model_with_shape.graph.value_info:
                        if (
                            hasattr(vi.type, "tensor_type")
                            and hasattr(vi.type.tensor_type, "elem_type")
                            and vi.type.tensor_type.elem_type != TensorProto.UNDEFINED
                            and vi.name
                        ):
                            vi_copy = ValueInfoProto()
                            vi_copy.CopyFrom(vi)
                            if hasattr(vi_copy.type.tensor_type, "shape"):
                                vi_copy.type.tensor_type.ClearField("shape")
                            name_vi[vi.name] = vi_copy
                    for vi in model.graph.value_info:
                        if vi.name in name_vi:
                            del name_vi[vi.name]
                    for vi in name_vi.values():
                        model.graph.value_info.append(vi)
            except Exception:
                logger.warning(
                    "Failed to run symbolic shape inference. Please file an issue in https://github.com/microsoft/onnxruntime."
                )

        parameters = {"disable_shape_infer": use_symbolic_shape_infer}
        parameters.update(
            {
                key: kwargs[key]
                for key in [
                    "keep_io_types",
                    "min_positive_val",
                    "max_finite_val",
                    "op_block_list",
                    "node_block_list",
                    "force_fp16_initializers",
                    "force_fp16_inputs",
                    "use_bfloat16_as_blocked_nodes_dtype",
                ]
                if key in kwargs
            }
        )

        fp16_model = convert_float_to_float16(model, **parameters)
        self.initialize(fp16_model)

        self.remove_cascaded_cast_nodes()

        self.remove_useless_cast_nodes()

    def create_node_name(self, op_type, name_prefix=None):
        """Create a unique node name that starts with a prefix (default is operator type).
           The name will not be duplicated with any name that generated or existed in current graphs.
        Args:
            op_type (str): operator type
            name_prefix (str, optional): prefix of node name. Defaults to None.

        Returns:
            str: node name
        """

        if name_prefix:
            prefix = name_prefix if name_prefix.endswith("_") else (name_prefix + "_")
        else:
            prefix = op_type + "_"

        suffix: int = 0
        if prefix in self._node_name_suffix:
            suffix = self._node_name_suffix[prefix] + 1
        else:
            # Check existed node name only once for a prefix
            # as we assume create_node_name is called for every new node in fusion.
            for node in self.nodes():
                if node.name and node.name.startswith(prefix):
                    try:
                        index = int(node.name[len(prefix) :])
                        suffix = max(index + 1, suffix)
                    except ValueError:
                        continue

        # Record the generated suffix so that we can avoid generating duplicated name.
        self._node_name_suffix[prefix] = suffix

        return prefix + str(suffix)

    def find_graph_input(self, input_name):
        for input in self.model.graph.input:
            if input.name == input_name:
                return input
        return None

    def find_graph_output(self, output_name):
        for output in self.model.graph.output:
            if output.name == output_name:
                return output
        return None

    def get_parent_subgraph_nodes(self, node, stop_nodes, output_name_to_node=None):
        if output_name_to_node is None:
            output_name_to_node = self.output_name_to_node()

        unique_nodes = []

        parents = self.get_parents(node, output_name_to_node)
        dq = deque(parents)
        while len(dq) > 0:
            current_node = dq.pop()
            if current_node in stop_nodes:
                continue

            if current_node not in unique_nodes:
                unique_nodes.append(current_node)

                for input in current_node.input:
                    if input in output_name_to_node:
                        dq.appendleft(output_name_to_node[input])

        return unique_nodes

    def get_graph_inputs(self, current_node, recursive=False):
        """
        Find graph inputs that linked to current node.
        """
        graph_inputs = []
        for input in current_node.input:
            if self.find_graph_input(input) and input not in graph_inputs:
                graph_inputs.append(input)

        if recursive:
            parent_nodes = self.get_parent_subgraph_nodes(current_node, [])
            for node in parent_nodes:
                for input in node.input:
                    if self.find_graph_input(input) and input not in graph_inputs:
                        graph_inputs.append(input)
        return graph_inputs

    @staticmethod
    def input_index(node_output, child_node):
        for index, input in enumerate(child_node.input):
            if input == node_output:
                return index
        return -1

    def remove_unused_constant(self):
        input_name_to_nodes = self.input_name_to_nodes()

        # remove unused constant
        unused_nodes = []
        nodes = self.nodes()
        for node in nodes:
            if node.op_type == "Constant" and node.output[0] not in input_name_to_nodes:
                unused_nodes.append(node)

        self.remove_nodes(unused_nodes)

        if len(unused_nodes) > 0:
            logger.debug(f"Removed unused constant nodes: {len(unused_nodes)}")

    def _get_subgraph_inputs_of_node(self, node):
        """
        Get inputs to all nodes in all subgraphs of a node
        """
        # Note: This function only handles one-level subgraphs of child nodes.
        subgraph_nodes_inputs = set()
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH:
                child_nodes = attr.g.node
                for child_node in child_nodes:
                    subgraph_nodes_inputs.update(child_node.input)
        return subgraph_nodes_inputs

    def _get_subgraph_nodes_and_inputs(self, ops_with_graph_attrs):
        """
        Get input names to all nodes in all subgraphs where subgraphs are
        graph attributes of a node in the main graph
        """
        subgraph_nodes = list(filter(lambda node: node.op_type in ops_with_graph_attrs, self.model.graph.node))
        subgraph_nodes_inputs = set()
        for parent_node in subgraph_nodes:
            subgraph_inputs_of_parent_node = self._get_subgraph_inputs_of_node(parent_node)
            subgraph_nodes_inputs.update(subgraph_inputs_of_parent_node)
        return subgraph_nodes, subgraph_nodes_inputs

    def prune_graph(self, outputs=None, allow_remove_graph_inputs=True):
        """
        Prune graph to keep only required outputs. It removes unnecessary nodes that are not linked
        (directly or indirectly) to any required output.

        There is also an option to remove graph inputs that are not used to generate any required output.

        Args:
            outputs (list): a list of graph outputs to retain. If it is None, all graph outputs will be kept.
            allow_remove_graph_inputs (bool): allow remove graph inputs.
        """

        keep_outputs = [output.name for output in self.model.graph.output] if outputs is None else outputs

        input_name_to_nodes_for_main_graph = self.input_name_to_nodes(exclude_subgraphs=True)
        output_name_to_node = self.output_name_to_node()

        def get_first_output(node):
            if node.output[0]:
                return node.output[0]
            return next(iter([o for o in node.output if o]), None)

        if len(self.graphs()) > 1:
            # Get input names for all nodes in all subgraphs
            subgraph_nodes, subgraph_nodes_inputs = self._get_subgraph_nodes_and_inputs(
                ops_with_graph_attrs={"Loop", "Scan", "If"}
            )
            if len(subgraph_nodes) == 0:
                # TODO: support other ops such as `BeamSearch` that have subgraphs as op attributes
                logger.debug("Skip prune_graph since graph has subgraph")
                return

            # For graphs with subgraphs, add dangling outputs from parent graph nodes to list of outputs to keep
            for node in self.model.graph.node:
                # TODO: This for-loop logic currently assumes that Loop/Scan/If nodes will not be
                # pruned because their subgraphs are needed for computations. This might not be
                # true in all cases.
                if node in subgraph_nodes:
                    continue

                # Check if node output is an input of a subgraph node and not an input to a node in the main graph
                for output in node.output:
                    if output in subgraph_nodes_inputs and output not in input_name_to_nodes_for_main_graph:
                        keep_outputs += [output]

        # Keep track of nodes to keep. The key is first output of node, and the value is the node.
        output_to_node = {}

        # Start from graph outputs, and find parent nodes recursively, and add nodes to the output_to_node dictionary.
        dq = deque()
        for output in keep_outputs:
            if output in output_name_to_node:
                dq.append(output_name_to_node[output])
        while len(dq) > 0:
            node = dq.pop()
            first_output = get_first_output(node)
            if first_output and (first_output not in output_to_node):
                output_to_node[first_output] = node
                for name in node.input:
                    if len(name) > 0 and (name in output_name_to_node) and (name not in output_to_node):
                        dq.appendleft(output_name_to_node[name])

        # Keep only those nodes in the output_to_node dictionary.
        nodes_to_keep = []
        num_nodes_removed = 0
        for node in self.model.graph.node:
            first_output = get_first_output(node)
            kept_node = output_to_node.get(first_output)

            # Need to double check the node since fused node might reuse output name of some nodes to be removed.
            # It is slow to compare whole node, so we compare op_type first to avoid comparing node in most cases.
            if kept_node and kept_node.op_type == node.op_type and kept_node == node:
                nodes_to_keep.append(node)
            else:
                num_nodes_removed += 1

        self.all_graphs = (
            None  # to prevent pass-by-copy after ClearField(), forces the use of pass-by-reference instead
        )
        self.model.graph.ClearField("node")
        self.model.graph.node.extend(nodes_to_keep)

        # Remove graph outputs not in list
        output_to_remove = []
        if outputs is not None:
            for output in self.model.graph.output:
                if output.name not in outputs:
                    output_to_remove.append(output)
            for output in output_to_remove:
                self.model.graph.output.remove(output)

        # Remove graph inputs not used by any node.
        input_to_remove = []
        if allow_remove_graph_inputs:
            input_name_to_nodes = self.input_name_to_nodes()
            input_to_remove = [input for input in self.model.graph.input if input.name not in input_name_to_nodes]
            for name in input_to_remove:
                self.model.graph.input.remove(name)

        if input_to_remove or output_to_remove or num_nodes_removed > 0:
            removed = []
            if input_to_remove:
                removed.append(f"{len(input_to_remove)} inputs")
            if output_to_remove:
                removed.append(f"{len(output_to_remove)} outputs")
            if num_nodes_removed > 0:
                removed.append(f"{num_nodes_removed} nodes")
            logger.info("Removed %s", ", ".join(removed))

        self.update_graph()

    def update_graph(self, verbose=False, allow_remove_graph_inputs=False):
        graph = self.model.graph

        remaining_input_names = set()
        for node in graph.node:
            if node.op_type in ["Loop", "Scan", "If"]:
                # Add input names of nodes in subgraphs
                subgraph_inputs_of_node = self._get_subgraph_inputs_of_node(node)
                remaining_input_names.update(subgraph_inputs_of_node)

            if node.op_type != "Constant":
                remaining_input_names.update(node.input)
        if verbose:
            logger.debug(f"remaining input names: {remaining_input_names}")

        # remove graph input that is not used
        inputs_to_remove = []
        if allow_remove_graph_inputs:
            for input in graph.input:
                if input.name not in remaining_input_names:
                    inputs_to_remove.append(input)
            for input in inputs_to_remove:
                graph.input.remove(input)

        names_to_remove = [input.name for input in inputs_to_remove]
        logger.debug(f"remove {len(inputs_to_remove)} unused inputs: {names_to_remove}")

        # remove weights that are not used
        weights_to_remove = []
        weights_to_keep = []
        for initializer in graph.initializer:
            if initializer.name not in remaining_input_names and not self.find_graph_output(initializer.name):
                weights_to_remove.append(initializer)
            else:
                weights_to_keep.append(initializer.name)
        for initializer in weights_to_remove:
            graph.initializer.remove(initializer)

        names_to_remove = [initializer.name for initializer in weights_to_remove]
        logger.debug(f"remove {len(weights_to_remove)} unused initializers: {names_to_remove}")
        if verbose:
            logger.debug(f"remaining initializers:{weights_to_keep}")

        self.remove_unused_constant()

    def is_safe_to_fuse_nodes(self, nodes_to_remove, keep_outputs, input_name_to_nodes, output_name_to_node):
        for node_to_remove in nodes_to_remove:
            for output_to_remove in node_to_remove.output:
                if output_to_remove in keep_outputs:
                    continue

                if output_to_remove in input_name_to_nodes:
                    for impacted_node in input_name_to_nodes[output_to_remove]:
                        if impacted_node not in nodes_to_remove:
                            logger.debug(
                                "it is not safe to remove nodes since output %s is used by %s",
                                output_to_remove,
                                impacted_node,
                            )
                            return False
        return True

    @staticmethod
    def graph_topological_sort(graph, is_deterministic=False):
        deps_set = set()  # dependency set of all node
        sorted_node_set = set()  # sorted node set
        sorted_nodes = []  # initialize sorted_nodes

        initializer_names = [init.name for init in graph.initializer]
        graph_input_names = [input.name for input in graph.input]
        input_names = initializer_names + graph_input_names

        if is_deterministic:
            input_names.sort()

        for input_name in input_names:
            deps_set.add(input_name)

        sorted_node_set_len = -1
        graph_nodes = graph.node if not is_deterministic else sorted(graph.node, key=lambda x: x.name)

        last_node_name = None
        while len(sorted_node_set) != len(graph_nodes):
            if len(sorted_node_set) == sorted_node_set_len:
                break
            sorted_node_set_len = len(sorted_node_set)
            for node_idx, node in enumerate(graph_nodes):
                if node_idx in sorted_node_set:
                    continue
                input_count = sum(1 for _ in node.input if _)
                if input_count == 0:
                    sorted_nodes.append(node)
                    sorted_node_set.add(node_idx)
                    for output in node.output:
                        if output:
                            deps_set.add(output)
                    continue
                failed = False
                for input_name in node.input:
                    if input_name and input_name not in deps_set:
                        failed = True
                        last_node_name = node.name
                if not failed:
                    sorted_nodes.append(node)
                    sorted_node_set.add(node_idx)
                    for output in node.output:
                        if output:
                            deps_set.add(output)
                else:
                    continue

        if len(sorted_node_set) != len(graph.node):
            raise RuntimeError(
                f"Graph is not a DAG: len(sorted_node_set)={len(sorted_node_set)}, len(graph.node)={len(graph.node)}, failed at node {last_node_name}"
            )

        graph.ClearField("node")
        graph.node.extend(sorted_nodes)

    def topological_sort(self, is_deterministic=False, dump_model_on_failure=False):
        # TODO: support graph_topological_sort() in subgraphs
        # for graph in self.graphs():
        #    self.graph_topological_sort(graph)
        try:
            OnnxModel.graph_topological_sort(self.model.graph, is_deterministic)
        except RuntimeError as e:
            if dump_model_on_failure:
                logger.info(
                    "Failed to sort graph in topological order. Dumping model to _topo_sort_failed.onnx for debugging."
                )
                OnnxModel.save(
                    self.model, "_topo_sort_failed.onnx", save_as_external_data=True, all_tensors_to_one_file=True
                )
            raise e

    @staticmethod
    def save(
        model,
        output_path,
        save_as_external_data=False,
        all_tensors_to_one_file=True,
        size_threshold=1024,
        convert_attribute=False,
    ):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Add ms domain if needed
        ms_opset = [opset for opset in model.opset_import if opset.domain == "com.microsoft"]
        # Check whether there is custom op in top level graph (our fusion is on top level right now).
        # May need to extend to subgraph if our fusion are extended to subgraphs.
        ms_node = [node for node in model.graph.node if node.domain == "com.microsoft"]
        if ms_node and not ms_opset:
            opset = model.opset_import.add()
            opset.version = 1
            opset.domain = "com.microsoft"

        if save_as_external_data:
            # Save model to external data, which is needed for model size > 2GB
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            external_data_path = output_path + ".data"
            location = Path(external_data_path).name if all_tensors_to_one_file else None

            if os.path.exists(output_path):
                logger.info(f"Delete the existing onnx file: {output_path}")
                os.remove(output_path)

            if all_tensors_to_one_file:
                if os.path.exists(external_data_path):
                    # Delete the external data file. Otherwise, data will be appended to existing file.
                    logger.info(f"Delete the existing external data file: {external_data_path}")
                    os.remove(external_data_path)
            else:
                if os.listdir(output_dir):
                    raise RuntimeError(f"Output directory ({output_dir}) for external data is not empty.")

            save_model(
                model,
                output_path,
                save_as_external_data=True,
                all_tensors_to_one_file=all_tensors_to_one_file,
                location=location,
                size_threshold=size_threshold,
                convert_attribute=convert_attribute,
            )
        else:
            save_model(model, output_path)

    def save_model_to_file(
        self,
        output_path,
        use_external_data_format=False,
        all_tensors_to_one_file=True,
        size_threshold=1024,
        convert_attribute=False,
    ):
        logger.info("Sort graphs in topological order")
        self.topological_sort()

        # Note: After the model is saved to another directory with external data,
        #       You need reload the onnx model if you want to read tensor from self.model object.
        #       It is because the base directory is not updated for self.model object so attempt to read tensor data
        #       might encounter error since external data cannot be located.
        OnnxModel.save(
            self.model,
            output_path,
            use_external_data_format,
            all_tensors_to_one_file,
            size_threshold,
            convert_attribute,
        )
        logger.info(f"Model saved to {output_path}")

    def get_graph_inputs_excluding_initializers(self):
        """
        Returns real graph inputs (excluding initializers from older onnx model).
        """
        graph_inputs = []
        for input in self.model.graph.input:
            if self.get_initializer(input.name) is None:
                graph_inputs.append(input)
        return graph_inputs

    def get_opset_version(self):
        """Get opset version of onnx domain

        Raises:
            RuntimeError: ONNX model has no opset for default domain.

        Returns:
            int: opset version of onnx domain.
        """
        for opset in self.model.opset_import:
            if opset.domain in ["", "ai.onnx"]:
                return opset.version
        raise RuntimeError("ONNX model has no opset for default domain")

    def get_operator_statistics(self, include_domain=False):
        """
        Returns node count of operators.
        """
        op_count = {}
        for node in self.nodes():
            op = (node.domain + ":" if include_domain and node.domain else "") + node.op_type
            op_count[op] = 1 if op not in op_count else (op_count[op] + 1)

        # Sorted by count in the descending order, then by key in alphabetical order.
        logger.info(f"Operators:{sorted(op_count.items(), key=lambda kv: (-kv[1], kv[0]))}")

        return op_count

    @staticmethod
    def to_data_hash(tensor: TensorProto, base_dir: str = "") -> int:
        """Converts a tensor def object to a hash for data comparison purposes.
        Args:
            tensor: a TensorProto object.
            base_dir: if external tensor exists, base_dir can help to find the path to it
        Returns:
            hash: a hash of the data.
        """
        if tensor.HasField("segment"):
            raise ValueError("Currently not supporting loading segments.")
        if tensor.data_type == TensorProto.UNDEFINED:
            raise TypeError("The element type in the input tensor is not defined.")
        tensor_dtype = tensor.data_type
        storage_field = helper.tensor_dtype_to_field(tensor_dtype)

        if tensor.data_type == TensorProto.STRING:
            utf8_strings = getattr(tensor, storage_field)
            return hash(tuple(s.decode("utf-8") for s in utf8_strings))
        # Load raw data from external tensor if it exists
        if uses_external_data(tensor):
            load_external_data_for_tensor(tensor, base_dir)
        if tensor.HasField("raw_data"):
            return hash(tensor.raw_data)
        else:
            np_data = numpy_helper.to_array(tensor)
            return hash(np_data.tobytes())

    @staticmethod
    def has_same_value(
        tensor1: TensorProto,
        tensor2: TensorProto,
        signature_cache1: dict | None = None,
        signature_cache2: dict | None = None,
    ) -> bool:
        """Returns True when two tensors have same value.
           Note that name can be different.

        Args:
            tensor1 (TensorProto): initializer 1
            tensor2 (TensorProto): initializer 2
            signature_cache1 (dict): Optional dictionary to store data signatures of tensor1 in order to speed up comparison.
            signature_cache2 (dict): Optional dictionary to store data signatures of tensor2 in order to speed up comparison.
        Returns:
            bool: True when two initializers has same value.
        """
        sig1 = (
            signature_cache1[tensor1.name]
            if signature_cache1 and tensor1.name in signature_cache1
            else OnnxModel.to_data_hash(tensor1)
        )
        sig2 = (
            signature_cache2[tensor2.name]
            if signature_cache2 and tensor2.name in signature_cache2
            else OnnxModel.to_data_hash(tensor2)
        )
        if signature_cache1 is not None:
            signature_cache1[tensor1.name] = sig1
        if signature_cache2 is not None:
            signature_cache2[tensor2.name] = sig2
        if sig1 == sig2 and tensor1.data_type == tensor2.data_type and tensor1.dims == tensor2.dims:
            # Same signature, now do the expensive check to confirm the data is the same
            return (numpy_helper.to_array(tensor1) == numpy_helper.to_array(tensor2)).all()

        return False

    def remove_initializer(self, tensor):
        for graph in self.graphs():
            if tensor in graph.initializer:
                graph.initializer.remove(tensor)
                return
        logger.warning("Failed to remove initializer %s", tensor)  # It might be a bug to hit this line.

    def remove_duplicated_initializer(self, cache: dict | None):
        """Remove initializers with duplicated values, and only keep the first one.
        It could help reduce size of models (like ALBert) with shared weights.
        If require_raw_data passed, method will only compare raw_data initializers to speed runtime
        Note: this function does not process subgraph.
        """
        if len(self.graphs()) > 1:
            logger.warning("remove_duplicated_initializer does not process subgraphs.")

        initializer_count = len(self.model.graph.initializer)

        same = [-1] * initializer_count
        for i in range(initializer_count - 1):
            if same[i] >= 0:
                continue
            for j in range(i + 1, initializer_count):
                if OnnxModel.has_same_value(
                    self.model.graph.initializer[i],
                    self.model.graph.initializer[j],
                    cache,
                    cache,
                ):
                    same[j] = i

        count = 0
        for i in range(initializer_count):
            if same[i] >= 0:
                count += 1
                self.replace_input_of_all_nodes(
                    self.model.graph.initializer[i].name,
                    self.model.graph.initializer[same[i]].name,
                )

        if count > 0:
            self.update_graph()
            print(f"Removed {count} initializers with duplicated value")

    def add_prefix_to_names(self, prefix: str):
        """Add prefix to initializer or intermediate outputs in graph. Main graph inputs and outputs are excluded.
        It could help avoid conflicting in name of node_args when merging two graphs.
        Note: this function does not process subgraph.
        """
        if len(self.graphs()) > 1:
            logger.warning("add_prefix_to_names does not process subgraphs.")

        # Exclude the names of inputs and outputs of main graph (but not subgraphs)
        # and empty names ("") as they have special meaning to denote missing optional inputs
        excluded = [i.name for i in self.model.graph.input] + [o.name for o in self.model.graph.output] + [""]

        for initializer in self.model.graph.initializer:
            if initializer.name not in excluded:
                if prefix + initializer.name not in excluded:
                    initializer.name = prefix + initializer.name

        for node in self.model.graph.node:
            # update name of node inputs
            for j in range(len(node.input)):
                if node.input[j] not in excluded:
                    if prefix + node.input[j] not in excluded:
                        node.input[j] = prefix + node.input[j]

            # update name of node outputs
            for j in range(len(node.output)):
                if node.output[j] not in excluded:
                    if prefix + node.output[j] not in excluded:
                        node.output[j] = prefix + node.output[j]

        for value_info in self.model.graph.value_info:
            if value_info.name not in excluded:
                value_info.name = prefix + value_info.name

    def clean_shape_infer(self):
        self.model.graph.ClearField("value_info")

    def use_float16(self):
        """Check whether the model uses float16"""
        queue = []  # queue for BFS
        queue.append(self.model.graph)
        while queue:
            sub_graphs = []
            for graph in queue:
                if not isinstance(graph, GraphProto):
                    continue

                for v in itertools.chain(graph.input, graph.output, graph.value_info):
                    if v.type.tensor_type.elem_type == TensorProto.FLOAT16:
                        return True
                    if v.type.HasField("sequence_type"):
                        if v.type.sequence_type.elem_type.tensor_type.elem_type == TensorProto.FLOAT16:
                            return True

                for t in graph.initializer:
                    if t.data_type == TensorProto.FLOAT16:
                        return True

                for node in graph.node:
                    if node.op_type == "Cast":
                        for attr in node.attribute:
                            if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                                return True

                    for attr in node.attribute:
                        if attr.type == AttributeProto.GRAPH:
                            sub_graphs.append(attr.g)

                        for g in attr.graphs:
                            sub_graphs.append(g)  # noqa: PERF402

                        if isinstance(attr.t, TensorProto) and attr.t.data_type == TensorProto.FLOAT16:
                            return True

                        for t in attr.tensors:
                            if isinstance(t, TensorProto) and t.data_type == TensorProto.FLOAT16:
                                return True

            queue = sub_graphs

        return False

    def change_graph_input_type(
        self,
        graph_input: ValueInfoProto,
        new_type: int,
    ):
        """Change graph input type, and add Cast node if needed.

        Args:
            graph_input (ValueInfoProto): input of the graph
            new_type (int): new data type like TensorProto.INT32.

        Returns:
            NodeProto: a new Cast node that added. None if Cast node is not added.
            List[NodeProto]: Cast nodes that have been removed.
        """
        assert isinstance(graph_input, ValueInfoProto)
        assert self.find_graph_input(graph_input.name)

        if graph_input.type.tensor_type.elem_type == int(new_type):
            return None, []

        graph = self.graph()
        new_cast_node = None
        nodes_to_remove = []

        input_name_to_nodes = self.input_name_to_nodes()
        if graph_input.name in input_name_to_nodes:
            nodes = input_name_to_nodes[graph_input.name]

            # For children that is not Cast node, insert a Cast node to convert int32 to original data type.
            nodes_not_cast = [node for node in nodes if node.op_type != "Cast"]
            if nodes_not_cast:
                node_name = self.create_node_name("Cast")
                output_name = node_name + "_" + graph_input.name
                new_value_info = graph.value_info.add()
                new_value_info.CopyFrom(graph_input)
                new_value_info.name = output_name
                new_cast_node = helper.make_node(
                    "Cast",
                    [graph_input.name],
                    [output_name],
                    to=int(graph_input.type.tensor_type.elem_type),
                    name=node_name,
                )
                graph.node.extend([new_cast_node])

                for node in nodes_not_cast:
                    OnnxModel.replace_node_input(node, graph_input.name, output_name)

            # For children that is Cast node, no need to insert Cast.
            # When the children is Cast to int32, we can remove that Cast node since input type is int32 now.
            nodes_cast = [node for node in nodes if node.op_type == "Cast"]
            for node in nodes_cast:
                if OnnxModel.get_node_attribute(node, "to") == int(new_type):
                    self.replace_input_of_all_nodes(node.output[0], graph_input.name)
                if not self.find_graph_output(node.output[0]):
                    nodes_to_remove.append(node)
            if nodes_to_remove:
                self.remove_nodes(nodes_to_remove)

        graph_input.type.tensor_type.elem_type = int(new_type)
        return new_cast_node, nodes_to_remove

    def change_graph_output_type(
        self,
        graph_output: ValueInfoProto,
        new_type: int,
    ):
        """Change graph input type, and add Cast node if needed.

        Args:
            graph_input (str | ValueInfoProto): output of the graph
            new_type (int): new data type.

        Returns:
            NodeProto: a new Cast node that added. None if Cast node is not added.
        """
        assert isinstance(graph_output, ValueInfoProto)
        assert self.find_graph_output(graph_output.name)

        if graph_output.type.tensor_type.elem_type == int(new_type):
            return None

        cast_node = None
        graph = self.graph()

        # Add a cast node
        node_name = self.create_node_name("Cast")
        input_name = node_name + "_" + graph_output.name
        self.replace_input_of_all_nodes(graph_output.name, input_name)
        new_value_info = graph.value_info.add()
        new_value_info.CopyFrom(graph_output)
        new_value_info.name = input_name
        cast_node = helper.make_node(
            "Cast",
            [input_name],
            [graph_output.name],
            to=int(new_type),
            name=node_name,
        )
        graph.node.extend([cast_node])
        graph_output.type.tensor_type.elem_type = int(new_type)
        return cast_node

    def rename_graph_output(self, old_name: str, new_name: str):
        if new_name in self.output_name_to_node():
            raise RuntimeError("{new_name} exists in graph")

        graph = self.graph()
        for output in graph.output:
            if output.name == old_name:
                logger.debug("replace output name from %s to %s", old_name, new_name)
                self.replace_input_of_all_nodes(old_name, new_name)
                self.replace_output_of_all_nodes(old_name, new_name)
                output.name = new_name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\relational.py ===
from __future__ import annotations

from .basic import Atom, Basic
from .coreerrors import LazyExceptionMessage
from .sorting import ordered
from .evalf import EvalfMixin
from .function import AppliedUndef
from .numbers import int_valued
from .singleton import S
from .sympify import _sympify, SympifyError
from .parameters import global_parameters
from .logic import fuzzy_bool, fuzzy_xor, fuzzy_and, fuzzy_not
from sympy.logic.boolalg import Boolean, BooleanAtom
from sympy.utilities.iterables import sift
from sympy.utilities.misc import filldedent
from sympy.utilities.exceptions import sympy_deprecation_warning


__all__ = (
    'Rel', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge',
    'Relational', 'Equality', 'Unequality', 'StrictLessThan', 'LessThan',
    'StrictGreaterThan', 'GreaterThan',
)

from .expr import Expr
from sympy.multipledispatch import dispatch
from .containers import Tuple
from .symbol import Symbol


def _nontrivBool(side):
    return isinstance(side, Boolean) and \
           not isinstance(side, Atom)


# Note, see issue 4986.  Ideally, we wouldn't want to subclass both Boolean
# and Expr.
# from .. import Expr


def _canonical(cond):
    # return a condition in which all relationals are canonical
    reps = {r: r.canonical for r in cond.atoms(Relational)}
    return cond.xreplace(reps)
    # XXX: AttributeError was being caught here but it wasn't triggered by any of
    # the tests so I've removed it...


def _canonical_coeff(rel):
    # return -2*x + 1 < 0 as x > 1/2
    # XXX make this part of Relational.canonical?
    rel = rel.canonical
    if not rel.is_Relational or rel.rhs.is_Boolean:
        return rel  # Eq(x, True)
    if not isinstance(rel.lhs, Expr):
        return rel.reversed  # e.g.: Eq(True, x) -> Eq(x, True)
    b, l = rel.lhs.as_coeff_Add(rational=True)
    m, lhs = l.as_coeff_Mul(rational=True)
    rhs = (rel.rhs - b)/m
    if m < 0:
        return rel.reversed.func(lhs, rhs)
    return rel.func(lhs, rhs)


class Relational(Boolean, EvalfMixin):
    """Base class for all relation types.

    Explanation
    ===========

    Subclasses of Relational should generally be instantiated directly, but
    Relational can be instantiated with a valid ``rop`` value to dispatch to
    the appropriate subclass.

    Parameters
    ==========

    rop : str or None
        Indicates what subclass to instantiate.  Valid values can be found
        in the keys of Relational.ValidRelationOperator.

    Examples
    ========

    >>> from sympy import Rel
    >>> from sympy.abc import x, y
    >>> Rel(y, x + x**2, '==')
    Eq(y, x**2 + x)

    A relation's type can be defined upon creation using ``rop``.
    The relation type of an existing expression can be obtained
    using its ``rel_op`` property.
    Here is a table of all the relation types, along with their
    ``rop`` and ``rel_op`` values:

    +---------------------+----------------------------+------------+
    |Relation             |``rop``                     |``rel_op``  |
    +=====================+============================+============+
    |``Equality``         |``==`` or ``eq`` or ``None``|``==``      |
    +---------------------+----------------------------+------------+
    |``Unequality``       |``!=`` or ``ne``            |``!=``      |
    +---------------------+----------------------------+------------+
    |``GreaterThan``      |``>=`` or ``ge``            |``>=``      |
    +---------------------+----------------------------+------------+
    |``LessThan``         |``<=`` or ``le``            |``<=``      |
    +---------------------+----------------------------+------------+
    |``StrictGreaterThan``|``>`` or ``gt``             |``>``       |
    +---------------------+----------------------------+------------+
    |``StrictLessThan``   |``<`` or ``lt``             |``<``       |
    +---------------------+----------------------------+------------+

    For example, setting ``rop`` to ``==`` produces an
    ``Equality`` relation, ``Eq()``.
    So does setting ``rop`` to ``eq``, or leaving ``rop`` unspecified.
    That is, the first three ``Rel()`` below all produce the same result.
    Using a ``rop`` from a different row in the table produces a
    different relation type.
    For example, the fourth ``Rel()`` below using ``lt`` for ``rop``
    produces a ``StrictLessThan`` inequality:

    >>> from sympy import Rel
    >>> from sympy.abc import x, y
    >>> Rel(y, x + x**2, '==')
        Eq(y, x**2 + x)
    >>> Rel(y, x + x**2, 'eq')
        Eq(y, x**2 + x)
    >>> Rel(y, x + x**2)
        Eq(y, x**2 + x)
    >>> Rel(y, x + x**2, 'lt')
        y < x**2 + x

    To obtain the relation type of an existing expression,
    get its ``rel_op`` property.
    For example, ``rel_op`` is ``==`` for the ``Equality`` relation above,
    and ``<`` for the strict less than inequality above:

    >>> from sympy import Rel
    >>> from sympy.abc import x, y
    >>> my_equality = Rel(y, x + x**2, '==')
    >>> my_equality.rel_op
        '=='
    >>> my_inequality = Rel(y, x + x**2, 'lt')
    >>> my_inequality.rel_op
        '<'

    """
    __slots__ = ()

    ValidRelationOperator: dict[str | None, type[Relational]] = {}

    is_Relational = True

    # ValidRelationOperator - Defined below, because the necessary classes
    #   have not yet been defined

    def __new__(cls, lhs, rhs, rop=None, **assumptions):
        # If called by a subclass, do nothing special and pass on to Basic.
        if cls is not Relational:
            return Basic.__new__(cls, lhs, rhs, **assumptions)

        # XXX: Why do this? There should be a separate function to make a
        # particular subclass of Relational from a string.
        #
        # If called directly with an operator, look up the subclass
        # corresponding to that operator and delegate to it
        cls = cls.ValidRelationOperator.get(rop, None)
        if cls is None:
            raise ValueError("Invalid relational operator symbol: %r" % rop)

        if not issubclass(cls, (Eq, Ne)):
            # validate that Booleans are not being used in a relational
            # other than Eq/Ne;
            # Note: Symbol is a subclass of Boolean but is considered
            # acceptable here.
            if any(map(_nontrivBool, (lhs, rhs))):
                raise TypeError(filldedent('''
                    A Boolean argument can only be used in
                    Eq and Ne; all other relationals expect
                    real expressions.
                '''))

        return cls(lhs, rhs, **assumptions)

    @property
    def lhs(self):
        """The left-hand side of the relation."""
        return self._args[0]

    @property
    def rhs(self):
        """The right-hand side of the relation."""
        return self._args[1]

    @property
    def reversed(self):
        """Return the relationship with sides reversed.

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x
        >>> Eq(x, 1)
        Eq(x, 1)
        >>> _.reversed
        Eq(1, x)
        >>> x < 1
        x < 1
        >>> _.reversed
        1 > x
        """
        ops = {Eq: Eq, Gt: Lt, Ge: Le, Lt: Gt, Le: Ge, Ne: Ne}
        a, b = self.args
        return Relational.__new__(ops.get(self.func, self.func), b, a)

    @property
    def reversedsign(self):
        """Return the relationship with signs reversed.

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x
        >>> Eq(x, 1)
        Eq(x, 1)
        >>> _.reversedsign
        Eq(-x, -1)
        >>> x < 1
        x < 1
        >>> _.reversedsign
        -x > -1
        """
        a, b = self.args
        if not (isinstance(a, BooleanAtom) or isinstance(b, BooleanAtom)):
            ops = {Eq: Eq, Gt: Lt, Ge: Le, Lt: Gt, Le: Ge, Ne: Ne}
            return Relational.__new__(ops.get(self.func, self.func), -a, -b)
        else:
            return self

    @property
    def negated(self):
        """Return the negated relationship.

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x
        >>> Eq(x, 1)
        Eq(x, 1)
        >>> _.negated
        Ne(x, 1)
        >>> x < 1
        x < 1
        >>> _.negated
        x >= 1

        Notes
        =====

        This works more or less identical to ``~``/``Not``. The difference is
        that ``negated`` returns the relationship even if ``evaluate=False``.
        Hence, this is useful in code when checking for e.g. negated relations
        to existing ones as it will not be affected by the `evaluate` flag.

        """
        ops = {Eq: Ne, Ge: Lt, Gt: Le, Le: Gt, Lt: Ge, Ne: Eq}
        # If there ever will be new Relational subclasses, the following line
        # will work until it is properly sorted out
        # return ops.get(self.func, lambda a, b, evaluate=False: ~(self.func(a,
        #      b, evaluate=evaluate)))(*self.args, evaluate=False)
        return Relational.__new__(ops.get(self.func), *self.args)

    @property
    def weak(self):
        """return the non-strict version of the inequality or self

        EXAMPLES
        ========

        >>> from sympy.abc import x
        >>> (x < 1).weak
        x <= 1
        >>> _.weak
        x <= 1
        """
        return self

    @property
    def strict(self):
        """return the strict version of the inequality or self

        EXAMPLES
        ========

        >>> from sympy.abc import x
        >>> (x <= 1).strict
        x < 1
        >>> _.strict
        x < 1
        """
        return self

    def _eval_evalf(self, prec):
        return self.func(*[s._evalf(prec) for s in self.args])

    @property
    def canonical(self):
        """Return a canonical form of the relational by putting a
        number on the rhs, canonically removing a sign or else
        ordering the args canonically. No other simplification is
        attempted.

        Examples
        ========

        >>> from sympy.abc import x, y
        >>> x < 2
        x < 2
        >>> _.reversed.canonical
        x < 2
        >>> (-y < x).canonical
        x > -y
        >>> (-y > x).canonical
        x < -y
        >>> (-y < -x).canonical
        x < y

        The canonicalization is recursively applied:

        >>> from sympy import Eq
        >>> Eq(x < y, y > x).canonical
        True
        """
        args = tuple([i.canonical if isinstance(i, Relational) else i for i in self.args])
        if args != self.args:
            r = self.func(*args)
            if not isinstance(r, Relational):
                return r
        else:
            r = self
        if r.rhs.is_number:
            if r.rhs.is_Number and r.lhs.is_Number and r.lhs > r.rhs:
                r = r.reversed
        elif r.lhs.is_number:
            r = r.reversed
        elif tuple(ordered(args)) != args:
            r = r.reversed

        LHS_CEMS = getattr(r.lhs, 'could_extract_minus_sign', None)
        RHS_CEMS = getattr(r.rhs, 'could_extract_minus_sign', None)

        if isinstance(r.lhs, BooleanAtom) or isinstance(r.rhs, BooleanAtom):
            return r

        # Check if first value has negative sign
        if LHS_CEMS and LHS_CEMS():
            return r.reversedsign
        elif not r.rhs.is_number and RHS_CEMS and RHS_CEMS():
            # Right hand side has a minus, but not lhs.
            # How does the expression with reversed signs behave?
            # This is so that expressions of the type
            # Eq(x, -y) and Eq(-x, y)
            # have the same canonical representation
            expr1, _ = ordered([r.lhs, -r.rhs])
            if expr1 != r.lhs:
                return r.reversed.reversedsign

        return r

    def equals(self, other, failing_expression=False):
        """Return True if the sides of the relationship are mathematically
        identical and the type of relationship is the same.
        If failing_expression is True, return the expression whose truth value
        was unknown."""
        if isinstance(other, Relational):
            if other in (self, self.reversed):
                return True
            a, b = self, other
            if a.func in (Eq, Ne) or b.func in (Eq, Ne):
                if a.func != b.func:
                    return False
                left, right = [i.equals(j,
                                        failing_expression=failing_expression)
                               for i, j in zip(a.args, b.args)]
                if left is True:
                    return right
                if right is True:
                    return left
                lr, rl = [i.equals(j, failing_expression=failing_expression)
                          for i, j in zip(a.args, b.reversed.args)]
                if lr is True:
                    return rl
                if rl is True:
                    return lr
                e = (left, right, lr, rl)
                if all(i is False for i in e):
                    return False
                for i in e:
                    if i not in (True, False):
                        return i
            else:
                if b.func != a.func:
                    b = b.reversed
                if a.func != b.func:
                    return False
                left = a.lhs.equals(b.lhs,
                                    failing_expression=failing_expression)
                if left is False:
                    return False
                right = a.rhs.equals(b.rhs,
                                     failing_expression=failing_expression)
                if right is False:
                    return False
                if left is True:
                    return right
                return left

    def _eval_simplify(self, **kwargs):
        from .add import Add
        from .expr import Expr
        r = self
        r = r.func(*[i.simplify(**kwargs) for i in r.args])
        if r.is_Relational:
            if not isinstance(r.lhs, Expr) or not isinstance(r.rhs, Expr):
                return r
            dif = r.lhs - r.rhs
            # replace dif with a valid Number that will
            # allow a definitive comparison with 0
            v = None
            if dif.is_comparable:
                v = dif.n(2)
                if any(i._prec == 1 for i in v.as_real_imag()):
                    rv, iv = [i.n(2) for i in dif.as_real_imag()]
                    v = rv + S.ImaginaryUnit*iv
            elif dif.equals(0):  # XXX this is expensive
                v = S.Zero
            if v is not None:
                r = r.func._eval_relation(v, S.Zero)
            r = r.canonical
            # If there is only one symbol in the expression,
            # try to write it on a simplified form
            free = list(filter(lambda x: x.is_real is not False, r.free_symbols))
            if len(free) == 1:
                try:
                    from sympy.solvers.solveset import linear_coeffs
                    x = free.pop()
                    dif = r.lhs - r.rhs
                    m, b = linear_coeffs(dif, x)
                    if m.is_zero is False:
                        if m.is_negative:
                            # Dividing with a negative number, so change order of arguments
                            # canonical will put the symbol back on the lhs later
                            r = r.func(-b / m, x)
                        else:
                            r = r.func(x, -b / m)
                    else:
                        r = r.func(b, S.Zero)
                except ValueError:
                    # maybe not a linear function, try polynomial
                    from sympy.polys.polyerrors import PolynomialError
                    from sympy.polys.polytools import gcd, Poly, poly
                    try:
                        p = poly(dif, x)
                        c = p.all_coeffs()
                        constant = c[-1]
                        c[-1] = 0
                        scale = gcd(c)
                        c = [ctmp / scale for ctmp in c]
                        r = r.func(Poly.from_list(c, x).as_expr(), -constant / scale)
                    except PolynomialError:
                        pass
            elif len(free) >= 2:
                try:
                    from sympy.solvers.solveset import linear_coeffs
                    from sympy.polys.polytools import gcd
                    free = list(ordered(free))
                    dif = r.lhs - r.rhs
                    m = linear_coeffs(dif, *free)
                    constant = m[-1]
                    del m[-1]
                    scale = gcd(m)
                    m = [mtmp / scale for mtmp in m]
                    nzm = list(filter(lambda f: f[0] != 0, list(zip(m, free))))
                    if scale.is_zero is False:
                        if constant != 0:
                            # lhs: expression, rhs: constant
                            newexpr = Add(*[i * j for i, j in nzm])
                            r = r.func(newexpr, -constant / scale)
                        else:
                            # keep first term on lhs
                            lhsterm = nzm[0][0] * nzm[0][1]
                            del nzm[0]
                            newexpr = Add(*[i * j for i, j in nzm])
                            r = r.func(lhsterm, -newexpr)

                    else:
                        r = r.func(constant, S.Zero)
                except ValueError:
                    pass
        # Did we get a simplified result?
        r = r.canonical
        measure = kwargs['measure']
        if measure(r) < kwargs['ratio'] * measure(self):
            return r
        else:
            return self

    def _eval_trigsimp(self, **opts):
        from sympy.simplify.trigsimp import trigsimp
        return self.func(trigsimp(self.lhs, **opts), trigsimp(self.rhs, **opts))

    def expand(self, **kwargs):
        args = (arg.expand(**kwargs) for arg in self.args)
        return self.func(*args)

    def __bool__(self) -> bool:
        raise TypeError(
            LazyExceptionMessage(
                lambda: f"cannot determine truth value of Relational: {self}"
            )
        )

    def _eval_as_set(self):
        # self is univariate and periodicity(self, x) in (0, None)
        from sympy.solvers.inequalities import solve_univariate_inequality
        from sympy.sets.conditionset import ConditionSet
        syms = self.free_symbols
        assert len(syms) == 1
        x = syms.pop()
        try:
            xset = solve_univariate_inequality(self, x, relational=False)
        except NotImplementedError:
            # solve_univariate_inequality raises NotImplementedError for
            # unsolvable equations/inequalities.
            xset = ConditionSet(x, self, S.Reals)
        return xset

    @property
    def binary_symbols(self):
        # override where necessary
        return set()


Rel = Relational


class Equality(Relational):
    """
    An equal relation between two objects.

    Explanation
    ===========

    Represents that two objects are equal.  If they can be easily shown
    to be definitively equal (or unequal), this will reduce to True (or
    False).  Otherwise, the relation is maintained as an unevaluated
    Equality object.  Use the ``simplify`` function on this object for
    more nontrivial evaluation of the equality relation.

    As usual, the keyword argument ``evaluate=False`` can be used to
    prevent any evaluation.

    Examples
    ========

    >>> from sympy import Eq, simplify, exp, cos
    >>> from sympy.abc import x, y
    >>> Eq(y, x + x**2)
    Eq(y, x**2 + x)
    >>> Eq(2, 5)
    False
    >>> Eq(2, 5, evaluate=False)
    Eq(2, 5)
    >>> _.doit()
    False
    >>> Eq(exp(x), exp(x).rewrite(cos))
    Eq(exp(x), sinh(x) + cosh(x))
    >>> simplify(_)
    True

    See Also
    ========

    sympy.logic.boolalg.Equivalent : for representing equality between two
        boolean expressions

    Notes
    =====

    Python treats 1 and True (and 0 and False) as being equal; SymPy
    does not. And integer will always compare as unequal to a Boolean:

    >>> Eq(True, 1), True == 1
    (False, True)

    This class is not the same as the == operator.  The == operator tests
    for exact structural equality between two expressions; this class
    compares expressions mathematically.

    If either object defines an ``_eval_Eq`` method, it can be used in place of
    the default algorithm.  If ``lhs._eval_Eq(rhs)`` or ``rhs._eval_Eq(lhs)``
    returns anything other than None, that return value will be substituted for
    the Equality.  If None is returned by ``_eval_Eq``, an Equality object will
    be created as usual.

    Since this object is already an expression, it does not respond to
    the method ``as_expr`` if one tries to create `x - y` from ``Eq(x, y)``.
    If ``eq = Eq(x, y)`` then write `eq.lhs - eq.rhs` to get ``x - y``.

    .. deprecated:: 1.5

       ``Eq(expr)`` with a single argument is a shorthand for ``Eq(expr, 0)``,
       but this behavior is deprecated and will be removed in a future version
       of SymPy.

    """
    rel_op = '=='

    __slots__ = ()

    is_Equality = True

    def __new__(cls, lhs, rhs, **options):
        evaluate = options.pop('evaluate', global_parameters.evaluate)
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
        if evaluate:
            val = is_eq(lhs, rhs)
            if val is None:
                return cls(lhs, rhs, evaluate=False)
            else:
                return _sympify(val)

        return Relational.__new__(cls, lhs, rhs)

    @classmethod
    def _eval_relation(cls, lhs, rhs):
        return _sympify(lhs == rhs)

    def _eval_rewrite_as_Add(self, L, R, evaluate=True, **kwargs):
        """
        return Eq(L, R) as L - R. To control the evaluation of
        the result set pass `evaluate=True` to give L - R;
        if `evaluate=None` then terms in L and R will not cancel
        but they will be listed in canonical order; otherwise
        non-canonical args will be returned. If one side is 0, the
        non-zero side will be returned.

        .. deprecated:: 1.13

           The method ``Eq.rewrite(Add)`` is deprecated.
           See :ref:`eq-rewrite-Add` for details.

        Examples
        ========

        >>> from sympy import Eq, Add
        >>> from sympy.abc import b, x
        >>> eq = Eq(x + b, x - b)
        >>> eq.rewrite(Add)  #doctest: +SKIP
        2*b
        >>> eq.rewrite(Add, evaluate=None).args  #doctest: +SKIP
        (b, b, x, -x)
        >>> eq.rewrite(Add, evaluate=False).args  #doctest: +SKIP
        (b, x, b, -x)
        """
        sympy_deprecation_warning("""
        Eq.rewrite(Add) is deprecated.

        For ``eq = Eq(a, b)`` use ``eq.lhs - eq.rhs`` to obtain
        ``a - b``.
        """,
            deprecated_since_version="1.13",
            active_deprecations_target="eq-rewrite-Add",
            stacklevel=5,
        )
        from .add import _unevaluated_Add, Add
        if L == 0:
            return R
        if R == 0:
            return L
        if evaluate:
            # allow cancellation of args
            return L - R
        args = Add.make_args(L) + Add.make_args(-R)
        if evaluate is None:
            # no cancellation, but canonical
            return _unevaluated_Add(*args)
        # no cancellation, not canonical
        return Add._from_args(args)

    @property
    def binary_symbols(self):
        if S.true in self.args or S.false in self.args:
            if self.lhs.is_Symbol:
                return {self.lhs}
            elif self.rhs.is_Symbol:
                return {self.rhs}
        return set()

    def _eval_simplify(self, **kwargs):
        # standard simplify
        e = super()._eval_simplify(**kwargs)
        if not isinstance(e, Equality):
            return e
        from .expr import Expr
        if not isinstance(e.lhs, Expr) or not isinstance(e.rhs, Expr):
            return e
        free = self.free_symbols
        if len(free) == 1:
            try:
                from .add import Add
                from sympy.solvers.solveset import linear_coeffs
                x = free.pop()
                m, b = linear_coeffs(
                    Add(e.lhs, -e.rhs, evaluate=False), x)
                if m.is_zero is False:
                    enew = e.func(x, -b / m)
                else:
                    enew = e.func(m * x, -b)
                measure = kwargs['measure']
                if measure(enew) <= kwargs['ratio'] * measure(e):
                    e = enew
            except ValueError:
                pass
        return e.canonical

    def integrate(self, *args, **kwargs):
        """See the integrate function in sympy.integrals"""
        from sympy.integrals.integrals import integrate
        return integrate(self, *args, **kwargs)

    def as_poly(self, *gens, **kwargs):
        '''Returns lhs-rhs as a Poly

        Examples
        ========

        >>> from sympy import Eq
        >>> from sympy.abc import x
        >>> Eq(x**2, 1).as_poly(x)
        Poly(x**2 - 1, x, domain='ZZ')
        '''
        return (self.lhs - self.rhs).as_poly(*gens, **kwargs)


Eq = Equality


class Unequality(Relational):
    """An unequal relation between two objects.

    Explanation
    ===========

    Represents that two objects are not equal.  If they can be shown to be
    definitively equal, this will reduce to False; if definitively unequal,
    this will reduce to True.  Otherwise, the relation is maintained as an
    Unequality object.

    Examples
    ========

    >>> from sympy import Ne
    >>> from sympy.abc import x, y
    >>> Ne(y, x+x**2)
    Ne(y, x**2 + x)

    See Also
    ========
    Equality

    Notes
    =====
    This class is not the same as the != operator.  The != operator tests
    for exact structural equality between two expressions; this class
    compares expressions mathematically.

    This class is effectively the inverse of Equality.  As such, it uses the
    same algorithms, including any available `_eval_Eq` methods.

    """
    rel_op = '!='

    __slots__ = ()

    def __new__(cls, lhs, rhs, **options):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)
        evaluate = options.pop('evaluate', global_parameters.evaluate)
        if evaluate:
            val = is_neq(lhs, rhs)
            if val is None:
                return cls(lhs, rhs, evaluate=False)
            else:
                return _sympify(val)

        return Relational.__new__(cls, lhs, rhs, **options)

    @classmethod
    def _eval_relation(cls, lhs, rhs):
        return _sympify(lhs != rhs)

    @property
    def binary_symbols(self):
        if S.true in self.args or S.false in self.args:
            if self.lhs.is_Symbol:
                return {self.lhs}
            elif self.rhs.is_Symbol:
                return {self.rhs}
        return set()

    def _eval_simplify(self, **kwargs):
        # simplify as an equality
        eq = Equality(*self.args)._eval_simplify(**kwargs)
        if isinstance(eq, Equality):
            # send back Ne with the new args
            return self.func(*eq.args)
        return eq.negated  # result of Ne is the negated Eq


Ne = Unequality


class _Inequality(Relational):
    """Internal base class for all *Than types.

    Each subclass must implement _eval_relation to provide the method for
    comparing two real numbers.

    """
    __slots__ = ()

    def __new__(cls, lhs, rhs, **options):

        try:
            lhs = _sympify(lhs)
            rhs = _sympify(rhs)
        except SympifyError:
            return NotImplemented

        evaluate = options.pop('evaluate', global_parameters.evaluate)
        if evaluate:
            for me in (lhs, rhs):
                if me.is_extended_real is False:
                    raise TypeError("Invalid comparison of non-real %s" % me)
                if me is S.NaN:
                    raise TypeError("Invalid NaN comparison")
            # First we invoke the appropriate inequality method of `lhs`
            # (e.g., `lhs.__lt__`).  That method will try to reduce to
            # boolean or raise an exception.  It may keep calling
            # superclasses until it reaches `Expr` (e.g., `Expr.__lt__`).
            # In some cases, `Expr` will just invoke us again (if neither it
            # nor a subclass was able to reduce to boolean or raise an
            # exception).  In that case, it must call us with
            # `evaluate=False` to prevent infinite recursion.
            return cls._eval_relation(lhs, rhs, **options)

        # make a "non-evaluated" Expr for the inequality
        return Relational.__new__(cls, lhs, rhs, **options)

    @classmethod
    def _eval_relation(cls, lhs, rhs, **options):
        val = cls._eval_fuzzy_relation(lhs, rhs)
        if val is None:
            return cls(lhs, rhs, evaluate=False)
        else:
            return _sympify(val)


class _Greater(_Inequality):
    """Not intended for general use

    _Greater is only used so that GreaterThan and StrictGreaterThan may
    subclass it for the .gts and .lts properties.

    """
    __slots__ = ()

    @property
    def gts(self):
        return self._args[0]

    @property
    def lts(self):
        return self._args[1]


class _Less(_Inequality):
    """Not intended for general use.

    _Less is only used so that LessThan and StrictLessThan may subclass it for
    the .gts and .lts properties.

    """
    __slots__ = ()

    @property
    def gts(self):
        return self._args[1]

    @property
    def lts(self):
        return self._args[0]


class GreaterThan(_Greater):
    r"""Class representations of inequalities.

    Explanation
    ===========

    The ``*Than`` classes represent inequal relationships, where the left-hand
    side is generally bigger or smaller than the right-hand side.  For example,
    the GreaterThan class represents an inequal relationship where the
    left-hand side is at least as big as the right side, if not bigger.  In
    mathematical notation:

    lhs $\ge$ rhs

    In total, there are four ``*Than`` classes, to represent the four
    inequalities:

    +-----------------+--------+
    |Class Name       | Symbol |
    +=================+========+
    |GreaterThan      | ``>=`` |
    +-----------------+--------+
    |LessThan         | ``<=`` |
    +-----------------+--------+
    |StrictGreaterThan| ``>``  |
    +-----------------+--------+
    |StrictLessThan   | ``<``  |
    +-----------------+--------+

    All classes take two arguments, lhs and rhs.

    +----------------------------+-----------------+
    |Signature Example           | Math Equivalent |
    +============================+=================+
    |GreaterThan(lhs, rhs)       |   lhs $\ge$ rhs |
    +----------------------------+-----------------+
    |LessThan(lhs, rhs)          |   lhs $\le$ rhs |
    +----------------------------+-----------------+
    |StrictGreaterThan(lhs, rhs) |   lhs $>$ rhs   |
    +----------------------------+-----------------+
    |StrictLessThan(lhs, rhs)    |   lhs $<$ rhs   |
    +----------------------------+-----------------+

    In addition to the normal .lhs and .rhs of Relations, ``*Than`` inequality
    objects also have the .lts and .gts properties, which represent the "less
    than side" and "greater than side" of the operator.  Use of .lts and .gts
    in an algorithm rather than .lhs and .rhs as an assumption of inequality
    direction will make more explicit the intent of a certain section of code,
    and will make it similarly more robust to client code changes:

    >>> from sympy import GreaterThan, StrictGreaterThan
    >>> from sympy import LessThan, StrictLessThan
    >>> from sympy import And, Ge, Gt, Le, Lt, Rel, S
    >>> from sympy.abc import x, y, z
    >>> from sympy.core.relational import Relational

    >>> e = GreaterThan(x, 1)
    >>> e
    x >= 1
    >>> '%s >= %s is the same as %s <= %s' % (e.gts, e.lts, e.lts, e.gts)
    'x >= 1 is the same as 1 <= x'

    Examples
    ========

    One generally does not instantiate these classes directly, but uses various
    convenience methods:

    >>> for f in [Ge, Gt, Le, Lt]:  # convenience wrappers
    ...     print(f(x, 2))
    x >= 2
    x > 2
    x <= 2
    x < 2

    Another option is to use the Python inequality operators (``>=``, ``>``,
    ``<=``, ``<``) directly.  Their main advantage over the ``Ge``, ``Gt``,
    ``Le``, and ``Lt`` counterparts, is that one can write a more
    "mathematical looking" statement rather than littering the math with
    oddball function calls.  However there are certain (minor) caveats of
    which to be aware (search for 'gotcha', below).

    >>> x >= 2
    x >= 2
    >>> _ == Ge(x, 2)
    True

    However, it is also perfectly valid to instantiate a ``*Than`` class less
    succinctly and less conveniently:

    >>> Rel(x, 1, ">")
    x > 1
    >>> Relational(x, 1, ">")
    x > 1

    >>> StrictGreaterThan(x, 1)
    x > 1
    >>> GreaterThan(x, 1)
    x >= 1
    >>> LessThan(x, 1)
    x <= 1
    >>> StrictLessThan(x, 1)
    x < 1

    Notes
    =====

    There are a couple of "gotchas" to be aware of when using Python's
    operators.

    The first is that what your write is not always what you get:

        >>> 1 < x
        x > 1

        Due to the order that Python parses a statement, it may
        not immediately find two objects comparable.  When ``1 < x``
        is evaluated, Python recognizes that the number 1 is a native
        number and that x is *not*.  Because a native Python number does
        not know how to compare itself with a SymPy object
        Python will try the reflective operation, ``x > 1`` and that is the
        form that gets evaluated, hence returned.

        If the order of the statement is important (for visual output to
        the console, perhaps), one can work around this annoyance in a
        couple ways:

        (1) "sympify" the literal before comparison

        >>> S(1) < x
        1 < x

        (2) use one of the wrappers or less succinct methods described
        above

        >>> Lt(1, x)
        1 < x
        >>> Relational(1, x, "<")
        1 < x

    The second gotcha involves writing equality tests between relationals
    when one or both sides of the test involve a literal relational:

        >>> e = x < 1; e
        x < 1
        >>> e == e  # neither side is a literal
        True
        >>> e == x < 1  # expecting True, too
        False
        >>> e != x < 1  # expecting False
        x < 1
        >>> x < 1 != x < 1  # expecting False or the same thing as before
        Traceback (most recent call last):
        ...
        TypeError: cannot determine truth value of Relational

        The solution for this case is to wrap literal relationals in
        parentheses:

        >>> e == (x < 1)
        True
        >>> e != (x < 1)
        False
        >>> (x < 1) != (x < 1)
        False

    The third gotcha involves chained inequalities not involving
    ``==`` or ``!=``. Occasionally, one may be tempted to write:

        >>> e = x < y < z
        Traceback (most recent call last):
        ...
        TypeError: symbolic boolean expression has no truth value.

        Due to an implementation detail or decision of Python [1]_,
        there is no way for SymPy to create a chained inequality with
        that syntax so one must use And:

        >>> e = And(x < y, y < z)
        >>> type( e )
        And
        >>> e
        (x < y) & (y < z)

        Although this can also be done with the '&' operator, it cannot
        be done with the 'and' operarator:

        >>> (x < y) & (y < z)
        (x < y) & (y < z)
        >>> (x < y) and (y < z)
        Traceback (most recent call last):
        ...
        TypeError: cannot determine truth value of Relational

    .. [1] This implementation detail is that Python provides no reliable
       method to determine that a chained inequality is being built.
       Chained comparison operators are evaluated pairwise, using "and"
       logic (see
       https://docs.python.org/3/reference/expressions.html#not-in). This
       is done in an efficient way, so that each object being compared
       is only evaluated once and the comparison can short-circuit. For
       example, ``1 > 2 > 3`` is evaluated by Python as ``(1 > 2) and (2
       > 3)``. The ``and`` operator coerces each side into a bool,
       returning the object itself when it short-circuits. The bool of
       the --Than operators will raise TypeError on purpose, because
       SymPy cannot determine the mathematical ordering of symbolic
       expressions. Thus, if we were to compute ``x > y > z``, with
       ``x``, ``y``, and ``z`` being Symbols, Python converts the
       statement (roughly) into these steps:

        (1) x > y > z
        (2) (x > y) and (y > z)
        (3) (GreaterThanObject) and (y > z)
        (4) (GreaterThanObject.__bool__()) and (y > z)
        (5) TypeError

       Because of the ``and`` added at step 2, the statement gets turned into a
       weak ternary statement, and the first object's ``__bool__`` method will
       raise TypeError.  Thus, creating a chained inequality is not possible.

           In Python, there is no way to override the ``and`` operator, or to
           control how it short circuits, so it is impossible to make something
           like ``x > y > z`` work.  There was a PEP to change this,
           :pep:`335`, but it was officially closed in March, 2012.

    """
    __slots__ = ()

    rel_op = '>='

    @classmethod
    def _eval_fuzzy_relation(cls, lhs, rhs):
        return is_ge(lhs, rhs)

    @property
    def strict(self):
        return Gt(*self.args)

Ge = GreaterThan


class LessThan(_Less):
    __doc__ = GreaterThan.__doc__
    __slots__ = ()

    rel_op = '<='

    @classmethod
    def _eval_fuzzy_relation(cls, lhs, rhs):
        return is_le(lhs, rhs)

    @property
    def strict(self):
        return Lt(*self.args)

Le = LessThan


class StrictGreaterThan(_Greater):
    __doc__ = GreaterThan.__doc__
    __slots__ = ()

    rel_op = '>'

    @classmethod
    def _eval_fuzzy_relation(cls, lhs, rhs):
        return is_gt(lhs, rhs)

    @property
    def weak(self):
        return Ge(*self.args)


Gt = StrictGreaterThan


class StrictLessThan(_Less):
    __doc__ = GreaterThan.__doc__
    __slots__ = ()

    rel_op = '<'

    @classmethod
    def _eval_fuzzy_relation(cls, lhs, rhs):
        return is_lt(lhs, rhs)

    @property
    def weak(self):
        return Le(*self.args)

Lt = StrictLessThan

# A class-specific (not object-specific) data item used for a minor speedup.
# It is defined here, rather than directly in the class, because the classes
# that it references have not been defined until now (e.g. StrictLessThan).
Relational.ValidRelationOperator = {
    None: Equality,
    '==': Equality,
    'eq': Equality,
    '!=': Unequality,
    '<>': Unequality,
    'ne': Unequality,
    '>=': GreaterThan,
    'ge': GreaterThan,
    '<=': LessThan,
    'le': LessThan,
    '>': StrictGreaterThan,
    'gt': StrictGreaterThan,
    '<': StrictLessThan,
    'lt': StrictLessThan,
}


def _n2(a, b):
    """Return (a - b).evalf(2) if a and b are comparable, else None.
    This should only be used when a and b are already sympified.
    """
    # /!\ it is very important (see issue 8245) not to
    # use a re-evaluated number in the calculation of dif
    if a.is_comparable and b.is_comparable:
        dif = (a - b).evalf(2)
        if dif.is_comparable:
            return dif


@dispatch(Expr, Expr)
def _eval_is_ge(lhs, rhs):
    return None


@dispatch(Basic, Basic)
def _eval_is_eq(lhs, rhs):
    return None


@dispatch(Tuple, Expr) # type: ignore
def _eval_is_eq(lhs, rhs):  # noqa:F811
    return False


@dispatch(Tuple, AppliedUndef) # type: ignore
def _eval_is_eq(lhs, rhs):  # noqa:F811
    return None


@dispatch(Tuple, Symbol) # type: ignore
def _eval_is_eq(lhs, rhs):  # noqa:F811
    return None


@dispatch(Tuple, Tuple) # type: ignore
def _eval_is_eq(lhs, rhs):  # noqa:F811
    if len(lhs) != len(rhs):
        return False

    return fuzzy_and(fuzzy_bool(is_eq(s, o)) for s, o in zip(lhs, rhs))


def is_lt(lhs, rhs, assumptions=None):
    """Fuzzy bool for lhs is strictly less than rhs.

    See the docstring for :func:`~.is_ge` for more.
    """
    return fuzzy_not(is_ge(lhs, rhs, assumptions))


def is_gt(lhs, rhs, assumptions=None):
    """Fuzzy bool for lhs is strictly greater than rhs.

    See the docstring for :func:`~.is_ge` for more.
    """
    return fuzzy_not(is_le(lhs, rhs, assumptions))


def is_le(lhs, rhs, assumptions=None):
    """Fuzzy bool for lhs is less than or equal to rhs.

    See the docstring for :func:`~.is_ge` for more.
    """
    return is_ge(rhs, lhs, assumptions)


def is_ge(lhs, rhs, assumptions=None):
    """
    Fuzzy bool for *lhs* is greater than or equal to *rhs*.

    Parameters
    ==========

    lhs : Expr
        The left-hand side of the expression, must be sympified,
        and an instance of expression. Throws an exception if
        lhs is not an instance of expression.

    rhs : Expr
        The right-hand side of the expression, must be sympified
        and an instance of expression. Throws an exception if
        lhs is not an instance of expression.

    assumptions: Boolean, optional
        Assumptions taken to evaluate the inequality.

    Returns
    =======

    ``True`` if *lhs* is greater than or equal to *rhs*, ``False`` if *lhs*
    is less than *rhs*, and ``None`` if the comparison between *lhs* and
    *rhs* is indeterminate.

    Explanation
    ===========

    This function is intended to give a relatively fast determination and
    deliberately does not attempt slow calculations that might help in
    obtaining a determination of True or False in more difficult cases.

    The four comparison functions ``is_le``, ``is_lt``, ``is_ge``, and ``is_gt`` are
    each implemented in terms of ``is_ge`` in the following way:

    is_ge(x, y) := is_ge(x, y)
    is_le(x, y) := is_ge(y, x)
    is_lt(x, y) := fuzzy_not(is_ge(x, y))
    is_gt(x, y) := fuzzy_not(is_ge(y, x))

    Therefore, supporting new type with this function will ensure behavior for
    other three functions as well.

    To maintain these equivalences in fuzzy logic it is important that in cases where
    either x or y is non-real all comparisons will give None.

    Examples
    ========

    >>> from sympy import S, Q
    >>> from sympy.core.relational import is_ge, is_le, is_gt, is_lt
    >>> from sympy.abc import x
    >>> is_ge(S(2), S(0))
    True
    >>> is_ge(S(0), S(2))
    False
    >>> is_le(S(0), S(2))
    True
    >>> is_gt(S(0), S(2))
    False
    >>> is_lt(S(2), S(0))
    False

    Assumptions can be passed to evaluate the quality which is otherwise
    indeterminate.

    >>> print(is_ge(x, S(0)))
    None
    >>> is_ge(x, S(0), assumptions=Q.positive(x))
    True

    New types can be supported by dispatching to ``_eval_is_ge``.

    >>> from sympy import Expr, sympify
    >>> from sympy.multipledispatch import dispatch
    >>> class MyExpr(Expr):
    ...     def __new__(cls, arg):
    ...         return super().__new__(cls, sympify(arg))
    ...     @property
    ...     def value(self):
    ...         return self.args[0]
    >>> @dispatch(MyExpr, MyExpr)
    ... def _eval_is_ge(a, b):
    ...     return is_ge(a.value, b.value)
    >>> a = MyExpr(1)
    >>> b = MyExpr(2)
    >>> is_ge(b, a)
    True
    >>> is_le(a, b)
    True
    """
    from sympy.assumptions.wrapper import AssumptionsWrapper, is_extended_nonnegative

    if not (isinstance(lhs, Expr) and isinstance(rhs, Expr)):
        raise TypeError("Can only compare inequalities with Expr")

    retval = _eval_is_ge(lhs, rhs)

    if retval is not None:
        return retval
    else:
        n2 = _n2(lhs, rhs)
        if n2 is not None:
            # use float comparison for infinity.
            # otherwise get stuck in infinite recursion
            if n2 in (S.Infinity, S.NegativeInfinity):
                n2 = float(n2)
            return n2 >= 0

        _lhs = AssumptionsWrapper(lhs, assumptions)
        _rhs = AssumptionsWrapper(rhs, assumptions)
        if _lhs.is_extended_real and _rhs.is_extended_real:
            if (_lhs.is_infinite and _lhs.is_extended_positive) or (_rhs.is_infinite and _rhs.is_extended_negative):
                return True
            diff = lhs - rhs
            if diff is not S.NaN:
                rv = is_extended_nonnegative(diff, assumptions)
                if rv is not None:
                    return rv


def is_neq(lhs, rhs, assumptions=None):
    """Fuzzy bool for lhs does not equal rhs.

    See the docstring for :func:`~.is_eq` for more.
    """
    return fuzzy_not(is_eq(lhs, rhs, assumptions))


def is_eq(lhs, rhs, assumptions=None):
    """
    Fuzzy bool representing mathematical equality between *lhs* and *rhs*.

    Parameters
    ==========

    lhs : Expr
        The left-hand side of the expression, must be sympified.

    rhs : Expr
        The right-hand side of the expression, must be sympified.

    assumptions: Boolean, optional
        Assumptions taken to evaluate the equality.

    Returns
    =======

    ``True`` if *lhs* is equal to *rhs*, ``False`` is *lhs* is not equal to *rhs*,
    and ``None`` if the comparison between *lhs* and *rhs* is indeterminate.

    Explanation
    ===========

    This function is intended to give a relatively fast determination and
    deliberately does not attempt slow calculations that might help in
    obtaining a determination of True or False in more difficult cases.

    :func:`~.is_neq` calls this function to return its value, so supporting
    new type with this function will ensure correct behavior for ``is_neq``
    as well.

    Examples
    ========

    >>> from sympy import Q, S
    >>> from sympy.core.relational import is_eq, is_neq
    >>> from sympy.abc import x
    >>> is_eq(S(0), S(0))
    True
    >>> is_neq(S(0), S(0))
    False
    >>> is_eq(S(0), S(2))
    False
    >>> is_neq(S(0), S(2))
    True

    Assumptions can be passed to evaluate the equality which is otherwise
    indeterminate.

    >>> print(is_eq(x, S(0)))
    None
    >>> is_eq(x, S(0), assumptions=Q.zero(x))
    True

    New types can be supported by dispatching to ``_eval_is_eq``.

    >>> from sympy import Basic, sympify
    >>> from sympy.multipledispatch import dispatch
    >>> class MyBasic(Basic):
    ...     def __new__(cls, arg):
    ...         return Basic.__new__(cls, sympify(arg))
    ...     @property
    ...     def value(self):
    ...         return self.args[0]
    ...
    >>> @dispatch(MyBasic, MyBasic)
    ... def _eval_is_eq(a, b):
    ...     return is_eq(a.value, b.value)
    ...
    >>> a = MyBasic(1)
    >>> b = MyBasic(1)
    >>> is_eq(a, b)
    True
    >>> is_neq(a, b)
    False

    """
    # here, _eval_Eq is only called for backwards compatibility
    # new code should use is_eq with multiple dispatch as
    # outlined in the docstring
    for side1, side2 in (lhs, rhs), (rhs, lhs):
        eval_func = getattr(side1, '_eval_Eq', None)
        if eval_func is not None:
            retval = eval_func(side2)
            if retval is not None:
                return retval

    retval = _eval_is_eq(lhs, rhs)
    if retval is not None:
        return retval

    if dispatch(type(lhs), type(rhs)) != dispatch(type(rhs), type(lhs)):
        retval = _eval_is_eq(rhs, lhs)
        if retval is not None:
            return retval

    # retval is still None, so go through the equality logic
    # If expressions have the same structure, they must be equal.
    if lhs == rhs:
        return True  # e.g. True == True
    elif all(isinstance(i, BooleanAtom) for i in (rhs, lhs)):
        return False  # True != False
    elif not (lhs.is_Symbol or rhs.is_Symbol) and (
        isinstance(lhs, Boolean) !=
        isinstance(rhs, Boolean)):
        return False  # only Booleans can equal Booleans

    from sympy.assumptions.wrapper import (AssumptionsWrapper,
        is_infinite, is_extended_real)
    from .add import Add

    _lhs = AssumptionsWrapper(lhs, assumptions)
    _rhs = AssumptionsWrapper(rhs, assumptions)

    if _lhs.is_infinite or _rhs.is_infinite:
        if fuzzy_xor([_lhs.is_infinite, _rhs.is_infinite]):
            return False
        if fuzzy_xor([_lhs.is_extended_real, _rhs.is_extended_real]):
            return False
        if fuzzy_and([_lhs.is_extended_real, _rhs.is_extended_real]):
            return fuzzy_xor([_lhs.is_extended_positive, fuzzy_not(_rhs.is_extended_positive)])

        # Try to split real/imaginary parts and equate them
        I = S.ImaginaryUnit

        def split_real_imag(expr):
            real_imag = lambda t: (
                'real' if is_extended_real(t, assumptions) else
                'imag' if is_extended_real(I*t, assumptions) else None)
            return sift(Add.make_args(expr), real_imag)

        lhs_ri = split_real_imag(lhs)
        if not lhs_ri[None]:
            rhs_ri = split_real_imag(rhs)
            if not rhs_ri[None]:
                eq_real = is_eq(Add(*lhs_ri['real']), Add(*rhs_ri['real']), assumptions)
                eq_imag = is_eq(I * Add(*lhs_ri['imag']), I * Add(*rhs_ri['imag']), assumptions)
                return fuzzy_and(map(fuzzy_bool, [eq_real, eq_imag]))

        from sympy.functions.elementary.complexes import arg
        # Compare e.g. zoo with 1+I*oo by comparing args
        arglhs = arg(lhs)
        argrhs = arg(rhs)
        # Guard against Eq(nan, nan) -> False
        if not (arglhs == S.NaN and argrhs == S.NaN):
            return fuzzy_bool(is_eq(arglhs, argrhs, assumptions))

    if all(isinstance(i, Expr) for i in (lhs, rhs)):
        # see if the difference evaluates
        dif = lhs - rhs
        _dif = AssumptionsWrapper(dif, assumptions)
        z = _dif.is_zero
        if z is not None:
            if z is False and _dif.is_commutative:  # issue 10728
                return False
            if z:
                return True

        # is_zero cannot help decide integer/rational with Float
        c, t = dif.as_coeff_Add()
        if c.is_Float:
            if int_valued(c):
                if t.is_integer is False:
                    return False
            elif t.is_rational is False:
                return False

        n2 = _n2(lhs, rhs)
        if n2 is not None:
            return _sympify(n2 == 0)

        # see if the ratio evaluates
        n, d = dif.as_numer_denom()
        rv = None
        _n = AssumptionsWrapper(n, assumptions)
        _d = AssumptionsWrapper(d, assumptions)
        if _n.is_zero:
            rv = _d.is_nonzero
        elif _n.is_finite:
            if _d.is_infinite:
                rv = True
            elif _n.is_zero is False:
                rv = _d.is_infinite
                if rv is None:
                    # if the condition that makes the denominator
                    # infinite does not make the original expression
                    # True then False can be returned
                    from sympy.simplify.simplify import clear_coefficients
                    l, r = clear_coefficients(d, S.Infinity)
                    args = [_.subs(l, r) for _ in (lhs, rhs)]
                    if args != [lhs, rhs]:
                        rv = fuzzy_bool(is_eq(*args, assumptions))
                        if rv is True:
                            rv = None
        elif any(is_infinite(a, assumptions) for a in Add.make_args(n)):
            # (inf or nan)/x != 0
            rv = False
        if rv is not None:
            return rv

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\decompositions.py ===
import copy

from sympy.core import S
from sympy.core.function import expand_mul
from sympy.functions.elementary.miscellaneous import Min, sqrt
from sympy.functions.elementary.complexes import sign

from .exceptions import NonSquareMatrixError, NonPositiveDefiniteMatrixError
from .utilities import _get_intermediate_simp, _iszero
from .determinant import _find_reasonable_pivot_naive


def _rank_decomposition(M, iszerofunc=_iszero, simplify=False):
    r"""Returns a pair of matrices (`C`, `F`) with matching rank
    such that `A = C F`.

    Parameters
    ==========

    iszerofunc : Function, optional
        A function used for detecting whether an element can
        act as a pivot.  ``lambda x: x.is_zero`` is used by default.

    simplify : Bool or Function, optional
        A function used to simplify elements when looking for a
        pivot. By default SymPy's ``simplify`` is used.

    Returns
    =======

    (C, F) : Matrices
        `C` and `F` are full-rank matrices with rank as same as `A`,
        whose product gives `A`.

        See Notes for additional mathematical details.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([
    ...     [1, 3, 1, 4],
    ...     [2, 7, 3, 9],
    ...     [1, 5, 3, 1],
    ...     [1, 2, 0, 8]
    ... ])
    >>> C, F = A.rank_decomposition()
    >>> C
    Matrix([
    [1, 3, 4],
    [2, 7, 9],
    [1, 5, 1],
    [1, 2, 8]])
    >>> F
    Matrix([
    [1, 0, -2, 0],
    [0, 1,  1, 0],
    [0, 0,  0, 1]])
    >>> C * F == A
    True

    Notes
    =====

    Obtaining `F`, an RREF of `A`, is equivalent to creating a
    product

    .. math::
        E_n E_{n-1} ... E_1 A = F

    where `E_n, E_{n-1}, \dots, E_1` are the elimination matrices or
    permutation matrices equivalent to each row-reduction step.

    The inverse of the same product of elimination matrices gives
    `C`:

    .. math::
        C = \left(E_n E_{n-1} \dots E_1\right)^{-1}

    It is not necessary, however, to actually compute the inverse:
    the columns of `C` are those from the original matrix with the
    same column indices as the indices of the pivot columns of `F`.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Rank_factorization

    .. [2] Piziak, R.; Odell, P. L. (1 June 1999).
        "Full Rank Factorization of Matrices".
        Mathematics Magazine. 72 (3): 193. doi:10.2307/2690882

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.rref
    """

    F, pivot_cols = M.rref(simplify=simplify, iszerofunc=iszerofunc,
            pivots=True)
    rank = len(pivot_cols)

    C = M.extract(range(M.rows), pivot_cols)
    F = F[:rank, :]

    return C, F


def _liupc(M):
    """Liu's algorithm, for pre-determination of the Elimination Tree of
    the given matrix, used in row-based symbolic Cholesky factorization.

    Examples
    ========

    >>> from sympy import SparseMatrix
    >>> S = SparseMatrix([
    ... [1, 0, 3, 2],
    ... [0, 0, 1, 0],
    ... [4, 0, 0, 5],
    ... [0, 6, 7, 0]])
    >>> S.liupc()
    ([[0], [], [0], [1, 2]], [4, 3, 4, 4])

    References
    ==========

    .. [1] Symbolic Sparse Cholesky Factorization using Elimination Trees,
           Jeroen Van Grondelle (1999)
           https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.39.7582
    """
    # Algorithm 2.4, p 17 of reference

    # get the indices of the elements that are non-zero on or below diag
    R = [[] for r in range(M.rows)]

    for r, c, _ in M.row_list():
        if c <= r:
            R[r].append(c)

    inf     = len(R)  # nothing will be this large
    parent  = [inf]*M.rows
    virtual = [inf]*M.rows

    for r in range(M.rows):
        for c in R[r][:-1]:
            while virtual[c] < r:
                t          = virtual[c]
                virtual[c] = r
                c          = t

            if virtual[c] == inf:
                parent[c] = virtual[c] = r

    return R, parent

def _row_structure_symbolic_cholesky(M):
    """Symbolic cholesky factorization, for pre-determination of the
    non-zero structure of the Cholesky factororization.

    Examples
    ========

    >>> from sympy import SparseMatrix
    >>> S = SparseMatrix([
    ... [1, 0, 3, 2],
    ... [0, 0, 1, 0],
    ... [4, 0, 0, 5],
    ... [0, 6, 7, 0]])
    >>> S.row_structure_symbolic_cholesky()
    [[0], [], [0], [1, 2]]

    References
    ==========

    .. [1] Symbolic Sparse Cholesky Factorization using Elimination Trees,
           Jeroen Van Grondelle (1999)
           https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.39.7582
    """

    R, parent = M.liupc()
    inf       = len(R)  # this acts as infinity
    Lrow      = copy.deepcopy(R)

    for k in range(M.rows):
        for j in R[k]:
            while j != inf and j != k:
                Lrow[k].append(j)
                j = parent[j]

        Lrow[k] = sorted(set(Lrow[k]))

    return Lrow


def _cholesky(M, hermitian=True):
    """Returns the Cholesky-type decomposition L of a matrix A
    such that L * L.H == A if hermitian flag is True,
    or L * L.T == A if hermitian is False.

    A must be a Hermitian positive-definite matrix if hermitian is True,
    or a symmetric matrix if it is False.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    >>> A.cholesky()
    Matrix([
    [ 5, 0, 0],
    [ 3, 3, 0],
    [-1, 1, 3]])
    >>> A.cholesky() * A.cholesky().T
    Matrix([
    [25, 15, -5],
    [15, 18,  0],
    [-5,  0, 11]])

    The matrix can have complex entries:

    >>> from sympy import I
    >>> A = Matrix(((9, 3*I), (-3*I, 5)))
    >>> A.cholesky()
    Matrix([
    [ 3, 0],
    [-I, 2]])
    >>> A.cholesky() * A.cholesky().H
    Matrix([
    [   9, 3*I],
    [-3*I,   5]])

    Non-hermitian Cholesky-type decomposition may be useful when the
    matrix is not positive-definite.

    >>> A = Matrix([[1, 2], [2, 1]])
    >>> L = A.cholesky(hermitian=False)
    >>> L
    Matrix([
    [1,         0],
    [2, sqrt(3)*I]])
    >>> L*L.T == A
    True

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.LDLdecomposition
    sympy.matrices.matrixbase.MatrixBase.LUdecomposition
    QRdecomposition
    """

    from .dense import MutableDenseMatrix

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if hermitian and not M.is_hermitian:
        raise ValueError("Matrix must be Hermitian.")
    if not hermitian and not M.is_symmetric():
        raise ValueError("Matrix must be symmetric.")

    L   = MutableDenseMatrix.zeros(M.rows, M.rows)

    if hermitian:
        for i in range(M.rows):
            for j in range(i):
                L[i, j] = ((1 / L[j, j])*(M[i, j] -
                    sum(L[i, k]*L[j, k].conjugate() for k in range(j))))

            Lii2 = (M[i, i] -
                sum(L[i, k]*L[i, k].conjugate() for k in range(i)))

            if Lii2.is_positive is False:
                raise NonPositiveDefiniteMatrixError(
                    "Matrix must be positive-definite")

            L[i, i] = sqrt(Lii2)

    else:
        for i in range(M.rows):
            for j in range(i):
                L[i, j] = ((1 / L[j, j])*(M[i, j] -
                    sum(L[i, k]*L[j, k] for k in range(j))))

            L[i, i] = sqrt(M[i, i] -
                sum(L[i, k]**2 for k in range(i)))

    return M._new(L)

def _cholesky_sparse(M, hermitian=True):
    """
    Returns the Cholesky decomposition L of a matrix A
    such that L * L.T = A

    A must be a square, symmetric, positive-definite
    and non-singular matrix

    Examples
    ========

    >>> from sympy import SparseMatrix
    >>> A = SparseMatrix(((25,15,-5),(15,18,0),(-5,0,11)))
    >>> A.cholesky()
    Matrix([
    [ 5, 0, 0],
    [ 3, 3, 0],
    [-1, 1, 3]])
    >>> A.cholesky() * A.cholesky().T == A
    True

    The matrix can have complex entries:

    >>> from sympy import I
    >>> A = SparseMatrix(((9, 3*I), (-3*I, 5)))
    >>> A.cholesky()
    Matrix([
    [ 3, 0],
    [-I, 2]])
    >>> A.cholesky() * A.cholesky().H
    Matrix([
    [   9, 3*I],
    [-3*I,   5]])

    Non-hermitian Cholesky-type decomposition may be useful when the
    matrix is not positive-definite.

    >>> A = SparseMatrix([[1, 2], [2, 1]])
    >>> L = A.cholesky(hermitian=False)
    >>> L
    Matrix([
    [1,         0],
    [2, sqrt(3)*I]])
    >>> L*L.T == A
    True

    See Also
    ========

    sympy.matrices.sparse.SparseMatrix.LDLdecomposition
    sympy.matrices.matrixbase.MatrixBase.LUdecomposition
    QRdecomposition
    """

    from .dense import MutableDenseMatrix

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if hermitian and not M.is_hermitian:
        raise ValueError("Matrix must be Hermitian.")
    if not hermitian and not M.is_symmetric():
        raise ValueError("Matrix must be symmetric.")

    dps       = _get_intermediate_simp(expand_mul, expand_mul)
    Crowstruc = M.row_structure_symbolic_cholesky()
    C         = MutableDenseMatrix.zeros(M.rows)

    for i in range(len(Crowstruc)):
        for j in Crowstruc[i]:
            if i != j:
                C[i, j] = M[i, j]
                summ    = 0

                for p1 in Crowstruc[i]:
                    if p1 < j:
                        for p2 in Crowstruc[j]:
                            if p2 < j:
                                if p1 == p2:
                                    if hermitian:
                                        summ += C[i, p1]*C[j, p1].conjugate()
                                    else:
                                        summ += C[i, p1]*C[j, p1]
                            else:
                                break
                        else:
                            break

                C[i, j] = dps((C[i, j] - summ) / C[j, j])

            else: # i == j
                C[j, j] = M[j, j]
                summ    = 0

                for k in Crowstruc[j]:
                    if k < j:
                        if hermitian:
                            summ += C[j, k]*C[j, k].conjugate()
                        else:
                            summ += C[j, k]**2
                    else:
                        break

                Cjj2 = dps(C[j, j] - summ)

                if hermitian and Cjj2.is_positive is False:
                    raise NonPositiveDefiniteMatrixError(
                        "Matrix must be positive-definite")

                C[j, j] = sqrt(Cjj2)

    return M._new(C)


def _LDLdecomposition(M, hermitian=True):
    """Returns the LDL Decomposition (L, D) of matrix A,
    such that L * D * L.H == A if hermitian flag is True, or
    L * D * L.T == A if hermitian is False.
    This method eliminates the use of square root.
    Further this ensures that all the diagonal entries of L are 1.
    A must be a Hermitian positive-definite matrix if hermitian is True,
    or a symmetric matrix otherwise.

    Examples
    ========

    >>> from sympy import Matrix, eye
    >>> A = Matrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    >>> L, D = A.LDLdecomposition()
    >>> L
    Matrix([
    [   1,   0, 0],
    [ 3/5,   1, 0],
    [-1/5, 1/3, 1]])
    >>> D
    Matrix([
    [25, 0, 0],
    [ 0, 9, 0],
    [ 0, 0, 9]])
    >>> L * D * L.T * A.inv() == eye(A.rows)
    True

    The matrix can have complex entries:

    >>> from sympy import I
    >>> A = Matrix(((9, 3*I), (-3*I, 5)))
    >>> L, D = A.LDLdecomposition()
    >>> L
    Matrix([
    [   1, 0],
    [-I/3, 1]])
    >>> D
    Matrix([
    [9, 0],
    [0, 4]])
    >>> L*D*L.H == A
    True

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.cholesky
    sympy.matrices.matrixbase.MatrixBase.LUdecomposition
    QRdecomposition
    """

    from .dense import MutableDenseMatrix

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if hermitian and not M.is_hermitian:
        raise ValueError("Matrix must be Hermitian.")
    if not hermitian and not M.is_symmetric():
        raise ValueError("Matrix must be symmetric.")

    D   = MutableDenseMatrix.zeros(M.rows, M.rows)
    L   = MutableDenseMatrix.eye(M.rows)

    if hermitian:
        for i in range(M.rows):
            for j in range(i):
                L[i, j] = (1 / D[j, j])*(M[i, j] - sum(
                    L[i, k]*L[j, k].conjugate()*D[k, k] for k in range(j)))

            D[i, i] = (M[i, i] -
                sum(L[i, k]*L[i, k].conjugate()*D[k, k] for k in range(i)))

            if D[i, i].is_positive is False:
                raise NonPositiveDefiniteMatrixError(
                    "Matrix must be positive-definite")

    else:
        for i in range(M.rows):
            for j in range(i):
                L[i, j] = (1 / D[j, j])*(M[i, j] - sum(
                    L[i, k]*L[j, k]*D[k, k] for k in range(j)))

            D[i, i] = M[i, i] - sum(L[i, k]**2*D[k, k] for k in range(i))

    return M._new(L), M._new(D)

def _LDLdecomposition_sparse(M, hermitian=True):
    """
    Returns the LDL Decomposition (matrices ``L`` and ``D``) of matrix
    ``A``, such that ``L * D * L.T == A``. ``A`` must be a square,
    symmetric, positive-definite and non-singular.

    This method eliminates the use of square root and ensures that all
    the diagonal entries of L are 1.

    Examples
    ========

    >>> from sympy import SparseMatrix
    >>> A = SparseMatrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    >>> L, D = A.LDLdecomposition()
    >>> L
    Matrix([
    [   1,   0, 0],
    [ 3/5,   1, 0],
    [-1/5, 1/3, 1]])
    >>> D
    Matrix([
    [25, 0, 0],
    [ 0, 9, 0],
    [ 0, 0, 9]])
    >>> L * D * L.T == A
    True

    """

    from .dense import MutableDenseMatrix

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if hermitian and not M.is_hermitian:
        raise ValueError("Matrix must be Hermitian.")
    if not hermitian and not M.is_symmetric():
        raise ValueError("Matrix must be symmetric.")

    dps       = _get_intermediate_simp(expand_mul, expand_mul)
    Lrowstruc = M.row_structure_symbolic_cholesky()
    L         = MutableDenseMatrix.eye(M.rows)
    D         = MutableDenseMatrix.zeros(M.rows, M.cols)

    for i in range(len(Lrowstruc)):
        for j in Lrowstruc[i]:
            if i != j:
                L[i, j] = M[i, j]
                summ    = 0

                for p1 in Lrowstruc[i]:
                    if p1 < j:
                        for p2 in Lrowstruc[j]:
                            if p2 < j:
                                if p1 == p2:
                                    if hermitian:
                                        summ += L[i, p1]*L[j, p1].conjugate()*D[p1, p1]
                                    else:
                                        summ += L[i, p1]*L[j, p1]*D[p1, p1]
                            else:
                                break
                    else:
                        break

                L[i, j] = dps((L[i, j] - summ) / D[j, j])

            else: # i == j
                D[i, i] = M[i, i]
                summ    = 0

                for k in Lrowstruc[i]:
                    if k < i:
                        if hermitian:
                            summ += L[i, k]*L[i, k].conjugate()*D[k, k]
                        else:
                            summ += L[i, k]**2*D[k, k]
                    else:
                        break

                D[i, i] = dps(D[i, i] - summ)

                if hermitian and D[i, i].is_positive is False:
                    raise NonPositiveDefiniteMatrixError(
                        "Matrix must be positive-definite")

    return M._new(L), M._new(D)


def _LUdecomposition(M, iszerofunc=_iszero, simpfunc=None, rankcheck=False):
    """Returns (L, U, perm) where L is a lower triangular matrix with unit
    diagonal, U is an upper triangular matrix, and perm is a list of row
    swap index pairs. If A is the original matrix, then
    ``A = (L*U).permuteBkwd(perm)``, and the row permutation matrix P such
    that $P A = L U$ can be computed by ``P = eye(A.rows).permuteFwd(perm)``.

    See documentation for LUCombined for details about the keyword argument
    rankcheck, iszerofunc, and simpfunc.

    Parameters
    ==========

    rankcheck : bool, optional
        Determines if this function should detect the rank
        deficiency of the matrixis and should raise a
        ``ValueError``.

    iszerofunc : function, optional
        A function which determines if a given expression is zero.

        The function should be a callable that takes a single
        SymPy expression and returns a 3-valued boolean value
        ``True``, ``False``, or ``None``.

        It is internally used by the pivot searching algorithm.
        See the notes section for a more information about the
        pivot searching algorithm.

    simpfunc : function or None, optional
        A function that simplifies the input.

        If this is specified as a function, this function should be
        a callable that takes a single SymPy expression and returns
        an another SymPy expression that is algebraically
        equivalent.

        If ``None``, it indicates that the pivot search algorithm
        should not attempt to simplify any candidate pivots.

        It is internally used by the pivot searching algorithm.
        See the notes section for a more information about the
        pivot searching algorithm.

    Examples
    ========

    >>> from sympy import Matrix
    >>> a = Matrix([[4, 3], [6, 3]])
    >>> L, U, _ = a.LUdecomposition()
    >>> L
    Matrix([
    [  1, 0],
    [3/2, 1]])
    >>> U
    Matrix([
    [4,    3],
    [0, -3/2]])

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.cholesky
    sympy.matrices.dense.DenseMatrix.LDLdecomposition
    QRdecomposition
    LUdecomposition_Simple
    LUdecompositionFF
    LUsolve
    """

    combined, p = M.LUdecomposition_Simple(iszerofunc=iszerofunc,
        simpfunc=simpfunc, rankcheck=rankcheck)

    # L is lower triangular ``M.rows x M.rows``
    # U is upper triangular ``M.rows x M.cols``
    # L has unit diagonal. For each column in combined, the subcolumn
    # below the diagonal of combined is shared by L.
    # If L has more columns than combined, then the remaining subcolumns
    # below the diagonal of L are zero.
    # The upper triangular portion of L and combined are equal.
    def entry_L(i, j):
        if i < j:
            # Super diagonal entry
            return M.zero
        elif i == j:
            return M.one
        elif j < combined.cols:
            return combined[i, j]

        # Subdiagonal entry of L with no corresponding
        # entry in combined
        return M.zero

    def entry_U(i, j):
        return M.zero if i > j else combined[i, j]

    L = M._new(combined.rows, combined.rows, entry_L)
    U = M._new(combined.rows, combined.cols, entry_U)

    return L, U, p

def _LUdecomposition_Simple(M, iszerofunc=_iszero, simpfunc=None,
        rankcheck=False):
    r"""Compute the PLU decomposition of the matrix.

    Parameters
    ==========

    rankcheck : bool, optional
        Determines if this function should detect the rank
        deficiency of the matrixis and should raise a
        ``ValueError``.

    iszerofunc : function, optional
        A function which determines if a given expression is zero.

        The function should be a callable that takes a single
        SymPy expression and returns a 3-valued boolean value
        ``True``, ``False``, or ``None``.

        It is internally used by the pivot searching algorithm.
        See the notes section for a more information about the
        pivot searching algorithm.

    simpfunc : function or None, optional
        A function that simplifies the input.

        If this is specified as a function, this function should be
        a callable that takes a single SymPy expression and returns
        an another SymPy expression that is algebraically
        equivalent.

        If ``None``, it indicates that the pivot search algorithm
        should not attempt to simplify any candidate pivots.

        It is internally used by the pivot searching algorithm.
        See the notes section for a more information about the
        pivot searching algorithm.

    Returns
    =======

    (lu, row_swaps) : (Matrix, list)
        If the original matrix is a $m, n$ matrix:

        *lu* is a $m, n$ matrix, which contains result of the
        decomposition in a compressed form. See the notes section
        to see how the matrix is compressed.

        *row_swaps* is a $m$-element list where each element is a
        pair of row exchange indices.

        ``A = (L*U).permute_backward(perm)``, and the row
        permutation matrix $P$ from the formula $P A = L U$ can be
        computed by ``P=eye(A.row).permute_forward(perm)``.

    Raises
    ======

    ValueError
        Raised if ``rankcheck=True`` and the matrix is found to
        be rank deficient during the computation.

    Notes
    =====

    About the PLU decomposition:

    PLU decomposition is a generalization of a LU decomposition
    which can be extended for rank-deficient matrices.

    It can further be generalized for non-square matrices, and this
    is the notation that SymPy is using.

    PLU decomposition is a decomposition of a $m, n$ matrix $A$ in
    the form of $P A = L U$ where

    * $L$ is a $m, m$ lower triangular matrix with unit diagonal
        entries.
    * $U$ is a $m, n$ upper triangular matrix.
    * $P$ is a $m, m$ permutation matrix.

    So, for a square matrix, the decomposition would look like:

    .. math::
        L = \begin{bmatrix}
        1 & 0 & 0 & \cdots & 0 \\
        L_{1, 0} & 1 & 0 & \cdots & 0 \\
        L_{2, 0} & L_{2, 1} & 1 & \cdots & 0 \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        L_{n-1, 0} & L_{n-1, 1} & L_{n-1, 2} & \cdots & 1
        \end{bmatrix}

    .. math::
        U = \begin{bmatrix}
        U_{0, 0} & U_{0, 1} & U_{0, 2} & \cdots & U_{0, n-1} \\
        0 & U_{1, 1} & U_{1, 2} & \cdots & U_{1, n-1} \\
        0 & 0 & U_{2, 2} & \cdots & U_{2, n-1} \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & 0 & \cdots & U_{n-1, n-1}
        \end{bmatrix}

    And for a matrix with more rows than the columns,
    the decomposition would look like:

    .. math::
        L = \begin{bmatrix}
        1 & 0 & 0 & \cdots & 0 & 0 & \cdots & 0 \\
        L_{1, 0} & 1 & 0 & \cdots & 0 & 0 & \cdots & 0 \\
        L_{2, 0} & L_{2, 1} & 1 & \cdots & 0 & 0 & \cdots & 0 \\
        \vdots & \vdots & \vdots & \ddots & \vdots & \vdots & \ddots
        & \vdots \\
        L_{n-1, 0} & L_{n-1, 1} & L_{n-1, 2} & \cdots & 1 & 0
        & \cdots & 0 \\
        L_{n, 0} & L_{n, 1} & L_{n, 2} & \cdots & L_{n, n-1} & 1
        & \cdots & 0 \\
        \vdots & \vdots & \vdots & \ddots & \vdots & \vdots
        & \ddots & \vdots \\
        L_{m-1, 0} & L_{m-1, 1} & L_{m-1, 2} & \cdots & L_{m-1, n-1}
        & 0 & \cdots & 1 \\
        \end{bmatrix}

    .. math::
        U = \begin{bmatrix}
        U_{0, 0} & U_{0, 1} & U_{0, 2} & \cdots & U_{0, n-1} \\
        0 & U_{1, 1} & U_{1, 2} & \cdots & U_{1, n-1} \\
        0 & 0 & U_{2, 2} & \cdots & U_{2, n-1} \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & 0 & \cdots & U_{n-1, n-1} \\
        0 & 0 & 0 & \cdots & 0 \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & 0 & \cdots & 0
        \end{bmatrix}

    Finally, for a matrix with more columns than the rows, the
    decomposition would look like:

    .. math::
        L = \begin{bmatrix}
        1 & 0 & 0 & \cdots & 0 \\
        L_{1, 0} & 1 & 0 & \cdots & 0 \\
        L_{2, 0} & L_{2, 1} & 1 & \cdots & 0 \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        L_{m-1, 0} & L_{m-1, 1} & L_{m-1, 2} & \cdots & 1
        \end{bmatrix}

    .. math::
        U = \begin{bmatrix}
        U_{0, 0} & U_{0, 1} & U_{0, 2} & \cdots & U_{0, m-1}
        & \cdots & U_{0, n-1} \\
        0 & U_{1, 1} & U_{1, 2} & \cdots & U_{1, m-1}
        & \cdots & U_{1, n-1} \\
        0 & 0 & U_{2, 2} & \cdots & U_{2, m-1}
        & \cdots & U_{2, n-1} \\
        \vdots & \vdots & \vdots & \ddots & \vdots
        & \cdots & \vdots \\
        0 & 0 & 0 & \cdots & U_{m-1, m-1}
        & \cdots & U_{m-1, n-1} \\
        \end{bmatrix}

    About the compressed LU storage:

    The results of the decomposition are often stored in compressed
    forms rather than returning $L$ and $U$ matrices individually.

    It may be less intiuitive, but it is commonly used for a lot of
    numeric libraries because of the efficiency.

    The storage matrix is defined as following for this specific
    method:

    * The subdiagonal elements of $L$ are stored in the subdiagonal
        portion of $LU$, that is $LU_{i, j} = L_{i, j}$ whenever
        $i > j$.
    * The elements on the diagonal of $L$ are all 1, and are not
        explicitly stored.
    * $U$ is stored in the upper triangular portion of $LU$, that is
        $LU_{i, j} = U_{i, j}$ whenever $i <= j$.
    * For a case of $m > n$, the right side of the $L$ matrix is
        trivial to store.
    * For a case of $m < n$, the below side of the $U$ matrix is
        trivial to store.

    So, for a square matrix, the compressed output matrix would be:

    .. math::
        LU = \begin{bmatrix}
        U_{0, 0} & U_{0, 1} & U_{0, 2} & \cdots & U_{0, n-1} \\
        L_{1, 0} & U_{1, 1} & U_{1, 2} & \cdots & U_{1, n-1} \\
        L_{2, 0} & L_{2, 1} & U_{2, 2} & \cdots & U_{2, n-1} \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        L_{n-1, 0} & L_{n-1, 1} & L_{n-1, 2} & \cdots & U_{n-1, n-1}
        \end{bmatrix}

    For a matrix with more rows than the columns, the compressed
    output matrix would be:

    .. math::
        LU = \begin{bmatrix}
        U_{0, 0} & U_{0, 1} & U_{0, 2} & \cdots & U_{0, n-1} \\
        L_{1, 0} & U_{1, 1} & U_{1, 2} & \cdots & U_{1, n-1} \\
        L_{2, 0} & L_{2, 1} & U_{2, 2} & \cdots & U_{2, n-1} \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        L_{n-1, 0} & L_{n-1, 1} & L_{n-1, 2} & \cdots
        & U_{n-1, n-1} \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        L_{m-1, 0} & L_{m-1, 1} & L_{m-1, 2} & \cdots
        & L_{m-1, n-1} \\
        \end{bmatrix}

    For a matrix with more columns than the rows, the compressed
    output matrix would be:

    .. math::
        LU = \begin{bmatrix}
        U_{0, 0} & U_{0, 1} & U_{0, 2} & \cdots & U_{0, m-1}
        & \cdots & U_{0, n-1} \\
        L_{1, 0} & U_{1, 1} & U_{1, 2} & \cdots & U_{1, m-1}
        & \cdots & U_{1, n-1} \\
        L_{2, 0} & L_{2, 1} & U_{2, 2} & \cdots & U_{2, m-1}
        & \cdots & U_{2, n-1} \\
        \vdots & \vdots & \vdots & \ddots & \vdots
        & \cdots & \vdots \\
        L_{m-1, 0} & L_{m-1, 1} & L_{m-1, 2} & \cdots & U_{m-1, m-1}
        & \cdots & U_{m-1, n-1} \\
        \end{bmatrix}

    About the pivot searching algorithm:

    When a matrix contains symbolic entries, the pivot search algorithm
    differs from the case where every entry can be categorized as zero or
    nonzero.
    The algorithm searches column by column through the submatrix whose
    top left entry coincides with the pivot position.
    If it exists, the pivot is the first entry in the current search
    column that iszerofunc guarantees is nonzero.
    If no such candidate exists, then each candidate pivot is simplified
    if simpfunc is not None.
    The search is repeated, with the difference that a candidate may be
    the pivot if ``iszerofunc()`` cannot guarantee that it is nonzero.
    In the second search the pivot is the first candidate that
    iszerofunc can guarantee is nonzero.
    If no such candidate exists, then the pivot is the first candidate
    for which iszerofunc returns None.
    If no such candidate exists, then the search is repeated in the next
    column to the right.
    The pivot search algorithm differs from the one in ``rref()``, which
    relies on ``_find_reasonable_pivot()``.
    Future versions of ``LUdecomposition_simple()`` may use
    ``_find_reasonable_pivot()``.

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.LUdecomposition
    LUdecompositionFF
    LUsolve
    """

    if rankcheck:
        # https://github.com/sympy/sympy/issues/9796
        pass

    if S.Zero in M.shape:
        # Define LU decomposition of a matrix with no entries as a matrix
        # of the same dimensions with all zero entries.
        return M.zeros(M.rows, M.cols), []

    dps       = _get_intermediate_simp()
    lu        = M.as_mutable()
    row_swaps = []

    pivot_col = 0

    for pivot_row in range(0, lu.rows - 1):
        # Search for pivot. Prefer entry that iszeropivot determines
        # is nonzero, over entry that iszeropivot cannot guarantee
        # is  zero.
        # XXX ``_find_reasonable_pivot`` uses slow zero testing. Blocked by bug #10279
        # Future versions of LUdecomposition_simple can pass iszerofunc and simpfunc
        # to _find_reasonable_pivot().
        # In pass 3 of _find_reasonable_pivot(), the predicate in ``if x.equals(S.Zero):``
        # calls sympy.simplify(), and not the simplification function passed in via
        # the keyword argument simpfunc.
        iszeropivot = True

        while pivot_col != M.cols and iszeropivot:
            sub_col = (lu[r, pivot_col] for r in range(pivot_row, M.rows))

            pivot_row_offset, pivot_value, is_assumed_non_zero, ind_simplified_pairs =\
                _find_reasonable_pivot_naive(sub_col, iszerofunc, simpfunc)

            iszeropivot = pivot_value is None

            if iszeropivot:
                # All candidate pivots in this column are zero.
                # Proceed to next column.
                pivot_col += 1

        if rankcheck and pivot_col != pivot_row:
            # All entries including and below the pivot position are
            # zero, which indicates that the rank of the matrix is
            # strictly less than min(num rows, num cols)
            # Mimic behavior of previous implementation, by throwing a
            # ValueError.
            raise ValueError("Rank of matrix is strictly less than"
                                " number of rows or columns."
                                " Pass keyword argument"
                                " rankcheck=False to compute"
                                " the LU decomposition of this matrix.")

        candidate_pivot_row = None if pivot_row_offset is None else pivot_row + pivot_row_offset

        if candidate_pivot_row is None and iszeropivot:
            # If candidate_pivot_row is None and iszeropivot is True
            # after pivot search has completed, then the submatrix
            # below and to the right of (pivot_row, pivot_col) is
            # all zeros, indicating that Gaussian elimination is
            # complete.
            return lu, row_swaps

        # Update entries simplified during pivot search.
        for offset, val in ind_simplified_pairs:
            lu[pivot_row + offset, pivot_col] = val

        if pivot_row != candidate_pivot_row:
            # Row swap book keeping:
            # Record which rows were swapped.
            # Update stored portion of L factor by multiplying L on the
            # left and right with the current permutation.
            # Swap rows of U.
            row_swaps.append([pivot_row, candidate_pivot_row])

            # Update L.
            lu[pivot_row, 0:pivot_row], lu[candidate_pivot_row, 0:pivot_row] = \
                lu[candidate_pivot_row, 0:pivot_row], lu[pivot_row, 0:pivot_row]

            # Swap pivot row of U with candidate pivot row.
            lu[pivot_row, pivot_col:lu.cols], lu[candidate_pivot_row, pivot_col:lu.cols] = \
                lu[candidate_pivot_row, pivot_col:lu.cols], lu[pivot_row, pivot_col:lu.cols]

        # Introduce zeros below the pivot by adding a multiple of the
        # pivot row to a row under it, and store the result in the
        # row under it.
        # Only entries in the target row whose index is greater than
        # start_col may be nonzero.
        start_col = pivot_col + 1

        for row in range(pivot_row + 1, lu.rows):
            # Store factors of L in the subcolumn below
            # (pivot_row, pivot_row).
            lu[row, pivot_row] = \
                dps(lu[row, pivot_col]/lu[pivot_row, pivot_col])

            # Form the linear combination of the pivot row and the current
            # row below the pivot row that zeros the entries below the pivot.
            # Employing slicing instead of a loop here raises
            # NotImplementedError: Cannot add Zero to MutableSparseMatrix
            # in sympy/matrices/tests/test_sparse.py.
            # c = pivot_row + 1 if pivot_row == pivot_col else pivot_col
            for c in range(start_col, lu.cols):
                lu[row, c] = dps(lu[row, c] - lu[row, pivot_row]*lu[pivot_row, c])

        if pivot_row != pivot_col:
            # matrix rank < min(num rows, num cols),
            # so factors of L are not stored directly below the pivot.
            # These entries are zero by construction, so don't bother
            # computing them.
            for row in range(pivot_row + 1, lu.rows):
                lu[row, pivot_col] = M.zero

        pivot_col += 1

        if pivot_col == lu.cols:
            # All candidate pivots are zero implies that Gaussian
            # elimination is complete.
            return lu, row_swaps

    if rankcheck:
        if iszerofunc(
                lu[Min(lu.rows, lu.cols) - 1, Min(lu.rows, lu.cols) - 1]):
            raise ValueError("Rank of matrix is strictly less than"
                                " number of rows or columns."
                                " Pass keyword argument"
                                " rankcheck=False to compute"
                                " the LU decomposition of this matrix.")

    return lu, row_swaps

def _LUdecompositionFF(M):
    """Compute a fraction-free LU decomposition.

    Returns 4 matrices P, L, D, U such that PA = L D**-1 U.
    If the elements of the matrix belong to some integral domain I, then all
    elements of L, D and U are guaranteed to belong to I.

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.LUdecomposition
    LUdecomposition_Simple
    LUsolve

    References
    ==========

    .. [1] W. Zhou & D.J. Jeffrey, "Fraction-free matrix factors: new forms
        for LU and QR factors". Frontiers in Computer Science in China,
        Vol 2, no. 1, pp. 67-80, 2008.
    """

    from sympy.matrices import SparseMatrix

    zeros    = SparseMatrix.zeros
    eye      = SparseMatrix.eye
    n, m     = M.rows, M.cols
    U, L, P  = M.as_mutable(), eye(n), eye(n)
    DD       = zeros(n, n)
    oldpivot = 1

    for k in range(n - 1):
        if U[k, k] == 0:
            for kpivot in range(k + 1, n):
                if U[kpivot, k]:
                    break
            else:
                raise ValueError("Matrix is not full rank")

            U[k, k:], U[kpivot, k:] = U[kpivot, k:], U[k, k:]
            L[k, :k], L[kpivot, :k] = L[kpivot, :k], L[k, :k]
            P[k, :], P[kpivot, :]   = P[kpivot, :], P[k, :]

        L [k, k] = Ukk = U[k, k]
        DD[k, k] = oldpivot * Ukk

        for i in range(k + 1, n):
            L[i, k] = Uik = U[i, k]

            for j in range(k + 1, m):
                U[i, j] = (Ukk * U[i, j] - U[k, j] * Uik) / oldpivot

            U[i, k] = 0

        oldpivot = Ukk

    DD[n - 1, n - 1] = oldpivot

    return P, L, DD, U

def _singular_value_decomposition(A):
    r"""Returns a Condensed Singular Value decomposition.

    Explanation
    ===========

    A Singular Value decomposition is a decomposition in the form $A = U \Sigma V^H$
    where

    - $U, V$ are column orthogonal matrix.
    - $\Sigma$ is a diagonal matrix, where the main diagonal contains singular
      values of matrix A.

    A column orthogonal matrix satisfies
    $\mathbb{I} = U^H U$ while a full orthogonal matrix satisfies
    relation $\mathbb{I} = U U^H = U^H U$ where $\mathbb{I}$ is an identity
    matrix with matching dimensions.

    For matrices which are not square or are rank-deficient, it is
    sufficient to return a column orthogonal matrix because augmenting
    them may introduce redundant computations.
    In condensed Singular Value Decomposition we only return column orthogonal
    matrices because of this reason

    If you want to augment the results to return a full orthogonal
    decomposition, you should use the following procedures.

    - Augment the $U, V$ matrices with columns that are orthogonal to every
      other columns and make it square.
    - Augment the $\Sigma$ matrix with zero rows to make it have the same
      shape as the original matrix.

    The procedure will be illustrated in the examples section.

    Examples
    ========

    we take a full rank matrix first:

    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2],[2,1]])
    >>> U, S, V = A.singular_value_decomposition()
    >>> U
    Matrix([
    [ sqrt(2)/2, sqrt(2)/2],
    [-sqrt(2)/2, sqrt(2)/2]])
    >>> S
    Matrix([
    [1, 0],
    [0, 3]])
    >>> V
    Matrix([
    [-sqrt(2)/2, sqrt(2)/2],
    [ sqrt(2)/2, sqrt(2)/2]])

    If a matrix if square and full rank both U, V
    are orthogonal in both directions

    >>> U * U.H
    Matrix([
    [1, 0],
    [0, 1]])
    >>> U.H * U
    Matrix([
    [1, 0],
    [0, 1]])

    >>> V * V.H
    Matrix([
    [1, 0],
    [0, 1]])
    >>> V.H * V
    Matrix([
    [1, 0],
    [0, 1]])
    >>> A == U * S * V.H
    True

    >>> C = Matrix([
    ...         [1, 0, 0, 0, 2],
    ...         [0, 0, 3, 0, 0],
    ...         [0, 0, 0, 0, 0],
    ...         [0, 2, 0, 0, 0],
    ...     ])
    >>> U, S, V = C.singular_value_decomposition()

    >>> V.H * V
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> V * V.H
    Matrix([
    [1/5, 0, 0, 0, 2/5],
    [  0, 1, 0, 0,   0],
    [  0, 0, 1, 0,   0],
    [  0, 0, 0, 0,   0],
    [2/5, 0, 0, 0, 4/5]])

    If you want to augment the results to be a full orthogonal
    decomposition, you should augment $V$ with an another orthogonal
    column.

    You are able to append an arbitrary standard basis that are linearly
    independent to every other columns and you can run the Gram-Schmidt
    process to make them augmented as orthogonal basis.

    >>> V_aug = V.row_join(Matrix([[0,0,0,0,1],
    ... [0,0,0,1,0]]).H)
    >>> V_aug = V_aug.QRdecomposition()[0]
    >>> V_aug
    Matrix([
    [0,   sqrt(5)/5, 0, -2*sqrt(5)/5, 0],
    [1,           0, 0,            0, 0],
    [0,           0, 1,            0, 0],
    [0,           0, 0,            0, 1],
    [0, 2*sqrt(5)/5, 0,    sqrt(5)/5, 0]])
    >>> V_aug.H * V_aug
    Matrix([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]])
    >>> V_aug * V_aug.H
    Matrix([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]])

    Similarly we augment U

    >>> U_aug = U.row_join(Matrix([0,0,1,0]))
    >>> U_aug = U_aug.QRdecomposition()[0]
    >>> U_aug
    Matrix([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0]])

    >>> U_aug.H * U_aug
    Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])
    >>> U_aug * U_aug.H
    Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])

    We add 2 zero columns and one row to S

    >>> S_aug = S.col_join(Matrix([[0,0,0]]))
    >>> S_aug = S_aug.row_join(Matrix([[0,0,0,0],
    ... [0,0,0,0]]).H)
    >>> S_aug
    Matrix([
    [2,       0, 0, 0, 0],
    [0, sqrt(5), 0, 0, 0],
    [0,       0, 3, 0, 0],
    [0,       0, 0, 0, 0]])



    >>> U_aug * S_aug * V_aug.H == C
    True

    """

    AH = A.H
    m, n = A.shape
    if m >= n:
        V, S = (AH * A).diagonalize()

        ranked = []
        for i, x in enumerate(S.diagonal()):
            if not x.is_zero:
                ranked.append(i)

        V = V[:, ranked]

        Singular_vals = [sqrt(S[i, i]) for i in range(S.rows) if i in ranked]

        S = S.diag(*Singular_vals)
        V, _ = V.QRdecomposition()
        U = A * V * S.inv()
    else:
        U, S = (A * AH).diagonalize()

        ranked = []
        for i, x in enumerate(S.diagonal()):
            if not x.is_zero:
                ranked.append(i)

        U = U[:, ranked]
        Singular_vals = [sqrt(S[i, i]) for i in range(S.rows) if i in ranked]

        S = S.diag(*Singular_vals)
        U, _ = U.QRdecomposition()
        V = AH * U * S.inv()

    return U, S, V

def _QRdecomposition_optional(M, normalize=True):
    def dot(u, v):
        return u.dot(v, hermitian=True)

    dps = _get_intermediate_simp(expand_mul, expand_mul)

    A = M.as_mutable()
    ranked = []

    Q = A
    R = A.zeros(A.cols)

    for j in range(A.cols):
        for i in range(j):
            if Q[:, i].is_zero_matrix:
                continue

            R[i, j] = dot(Q[:, i], Q[:, j]) / dot(Q[:, i], Q[:, i])
            R[i, j] = dps(R[i, j])
            Q[:, j] -= Q[:, i] * R[i, j]

        Q[:, j] = dps(Q[:, j])
        if Q[:, j].is_zero_matrix is not True:
            ranked.append(j)
            R[j, j] = M.one

    Q = Q.extract(range(Q.rows), ranked)
    R = R.extract(ranked, range(R.cols))

    if normalize:
        # Normalization
        for i in range(Q.cols):
            norm = Q[:, i].norm()
            Q[:, i] /= norm
            R[i, :] *= norm

    return M.__class__(Q), M.__class__(R)


def _QRdecomposition(M):
    r"""Returns a QR decomposition.

    Explanation
    ===========

    A QR decomposition is a decomposition in the form $A = Q R$
    where

    - $Q$ is a column orthogonal matrix.
    - $R$ is a upper triangular (trapezoidal) matrix.

    A column orthogonal matrix satisfies
    $\mathbb{I} = Q^H Q$ while a full orthogonal matrix satisfies
    relation $\mathbb{I} = Q Q^H = Q^H Q$ where $I$ is an identity
    matrix with matching dimensions.

    For matrices which are not square or are rank-deficient, it is
    sufficient to return a column orthogonal matrix because augmenting
    them may introduce redundant computations.
    And an another advantage of this is that you can easily inspect the
    matrix rank by counting the number of columns of $Q$.

    If you want to augment the results to return a full orthogonal
    decomposition, you should use the following procedures.

    - Augment the $Q$ matrix with columns that are orthogonal to every
      other columns and make it square.
    - Augment the $R$ matrix with zero rows to make it have the same
      shape as the original matrix.

    The procedure will be illustrated in the examples section.

    Examples
    ========

    A full rank matrix example:

    >>> from sympy import Matrix
    >>> A = Matrix([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
    >>> Q, R = A.QRdecomposition()
    >>> Q
    Matrix([
    [ 6/7, -69/175, -58/175],
    [ 3/7, 158/175,   6/175],
    [-2/7,    6/35,  -33/35]])
    >>> R
    Matrix([
    [14,  21, -14],
    [ 0, 175, -70],
    [ 0,   0,  35]])

    If the matrix is square and full rank, the $Q$ matrix becomes
    orthogonal in both directions, and needs no augmentation.

    >>> Q * Q.H
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> Q.H * Q
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

    >>> A == Q*R
    True

    A rank deficient matrix example:

    >>> A = Matrix([[12, -51, 0], [6, 167, 0], [-4, 24, 0]])
    >>> Q, R = A.QRdecomposition()
    >>> Q
    Matrix([
    [ 6/7, -69/175],
    [ 3/7, 158/175],
    [-2/7,    6/35]])
    >>> R
    Matrix([
    [14,  21, 0],
    [ 0, 175, 0]])

    QRdecomposition might return a matrix Q that is rectangular.
    In this case the orthogonality condition might be satisfied as
    $\mathbb{I} = Q.H*Q$ but not in the reversed product
    $\mathbb{I} = Q * Q.H$.

    >>> Q.H * Q
    Matrix([
    [1, 0],
    [0, 1]])
    >>> Q * Q.H
    Matrix([
    [27261/30625,   348/30625, -1914/6125],
    [  348/30625, 30589/30625,   198/6125],
    [ -1914/6125,    198/6125,   136/1225]])

    If you want to augment the results to be a full orthogonal
    decomposition, you should augment $Q$ with an another orthogonal
    column.

    You are able to append an identity matrix,
    and you can run the Gram-Schmidt
    process to make them augmented as orthogonal basis.

    >>> Q_aug = Q.row_join(Matrix.eye(3))
    >>> Q_aug = Q_aug.QRdecomposition()[0]
    >>> Q_aug
    Matrix([
    [ 6/7, -69/175, 58/175],
    [ 3/7, 158/175, -6/175],
    [-2/7,    6/35,  33/35]])
    >>> Q_aug.H * Q_aug
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> Q_aug * Q_aug.H
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

    Augmenting the $R$ matrix with zero row is straightforward.

    >>> R_aug = R.col_join(Matrix([[0, 0, 0]]))
    >>> R_aug
    Matrix([
    [14,  21, 0],
    [ 0, 175, 0],
    [ 0,   0, 0]])
    >>> Q_aug * R_aug == A
    True

    A zero matrix example:

    >>> from sympy import Matrix
    >>> A = Matrix.zeros(3, 4)
    >>> Q, R = A.QRdecomposition()

    They may return matrices with zero rows and columns.

    >>> Q
    Matrix(3, 0, [])
    >>> R
    Matrix(0, 4, [])
    >>> Q*R
    Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]])

    As the same augmentation rule described above, $Q$ can be augmented
    with columns of an identity matrix and $R$ can be augmented with
    rows of a zero matrix.

    >>> Q_aug = Q.row_join(Matrix.eye(3))
    >>> R_aug = R.col_join(Matrix.zeros(3, 4))
    >>> Q_aug * Q_aug.T
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> R_aug
    Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]])
    >>> Q_aug * R_aug == A
    True

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.cholesky
    sympy.matrices.dense.DenseMatrix.LDLdecomposition
    sympy.matrices.matrixbase.MatrixBase.LUdecomposition
    QRsolve
    """
    return _QRdecomposition_optional(M, normalize=True)

def _upper_hessenberg_decomposition(A):
    """Converts a matrix into Hessenberg matrix H.

    Returns 2 matrices H, P s.t.
    $P H P^{T} = A$, where H is an upper hessenberg matrix
    and P is an orthogonal matrix

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([
    ...     [1,2,3],
    ...     [-3,5,6],
    ...     [4,-8,9],
    ... ])
    >>> H, P = A.upper_hessenberg_decomposition()
    >>> H
    Matrix([
    [1,    6/5,    17/5],
    [5, 213/25, -134/25],
    [0, 216/25,  137/25]])
    >>> P
    Matrix([
    [1,    0,   0],
    [0, -3/5, 4/5],
    [0,  4/5, 3/5]])
    >>> P * H * P.H == A
    True


    References
    ==========

    .. [#] https://mathworld.wolfram.com/HessenbergDecomposition.html
    """

    M = A.as_mutable()

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")

    n = M.cols
    P = M.eye(n)
    H = M

    for j in range(n - 2):

        u = H[j + 1:, j]

        if u[1:, :].is_zero_matrix:
            continue

        if sign(u[0]) != 0:
            u[0] = u[0] + sign(u[0]) * u.norm()
        else:
            u[0] = u[0] + u.norm()

        v = u / u.norm()

        H[j + 1:, :] = H[j + 1:, :] - 2 * v * (v.H * H[j + 1:, :])
        H[:, j + 1:] = H[:, j + 1:] - (H[:, j + 1:] * (2 * v)) * v.H
        P[:, j + 1:] = P[:, j + 1:] - (P[:, j + 1:] * (2 * v)) * v.H

    return H, P

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\polynomial\polynomial.py ===
"""
=================================================
Power Series (:mod:`numpy.polynomial.polynomial`)
=================================================

This module provides a number of objects (mostly functions) useful for
dealing with polynomials, including a `Polynomial` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with polynomial objects is in
the docstring for its "parent" sub-package, `numpy.polynomial`).

Classes
-------
.. autosummary::
   :toctree: generated/

   Polynomial

Constants
---------
.. autosummary::
   :toctree: generated/

   polydomain
   polyzero
   polyone
   polyx

Arithmetic
----------
.. autosummary::
   :toctree: generated/

   polyadd
   polysub
   polymulx
   polymul
   polydiv
   polypow
   polyval
   polyval2d
   polyval3d
   polygrid2d
   polygrid3d

Calculus
--------
.. autosummary::
   :toctree: generated/

   polyder
   polyint

Misc Functions
--------------
.. autosummary::
   :toctree: generated/

   polyfromroots
   polyroots
   polyvalfromroots
   polyvander
   polyvander2d
   polyvander3d
   polycompanion
   polyfit
   polytrim
   polyline

See Also
--------
`numpy.polynomial`

"""
__all__ = [
    'polyzero', 'polyone', 'polyx', 'polydomain', 'polyline', 'polyadd',
    'polysub', 'polymulx', 'polymul', 'polydiv', 'polypow', 'polyval',
    'polyvalfromroots', 'polyder', 'polyint', 'polyfromroots', 'polyvander',
    'polyfit', 'polytrim', 'polyroots', 'Polynomial', 'polyval2d', 'polyval3d',
    'polygrid2d', 'polygrid3d', 'polyvander2d', 'polyvander3d',
    'polycompanion']

import numpy as np
import numpy.linalg as la
from numpy.lib.array_utils import normalize_axis_index

from . import polyutils as pu
from ._polybase import ABCPolyBase

polytrim = pu.trimcoef

#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Polynomial default domain.
polydomain = np.array([-1., 1.])

# Polynomial coefficients representing zero.
polyzero = np.array([0])

# Polynomial coefficients representing one.
polyone = np.array([1])

# Polynomial coefficients representing the identity x.
polyx = np.array([0, 1])

#
# Polynomial series functions
#


def polyline(off, scl):
    """
    Returns an array representing a linear polynomial.

    Parameters
    ----------
    off, scl : scalars
        The "y-intercept" and "slope" of the line, respectively.

    Returns
    -------
    y : ndarray
        This module's representation of the linear polynomial ``off +
        scl*x``.

    See Also
    --------
    numpy.polynomial.chebyshev.chebline
    numpy.polynomial.legendre.legline
    numpy.polynomial.laguerre.lagline
    numpy.polynomial.hermite.hermline
    numpy.polynomial.hermite_e.hermeline

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> P.polyline(1, -1)
    array([ 1, -1])
    >>> P.polyval(1, P.polyline(1, -1))  # should be 0
    0.0

    """
    if scl != 0:
        return np.array([off, scl])
    else:
        return np.array([off])


def polyfromroots(roots):
    """
    Generate a monic polynomial with given roots.

    Return the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    where the :math:`r_n` are the roots specified in `roots`.  If a zero has
    multiplicity n, then it must appear in `roots` n times. For instance,
    if 2 is a root of multiplicity three and 3 is a root of multiplicity 2,
    then `roots` looks something like [2, 2, 2, 3, 3]. The roots can appear
    in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * x + ... +  x^n

    The coefficient of the last term is 1 for monic polynomials in this
    form.

    Parameters
    ----------
    roots : array_like
        Sequence containing the roots.

    Returns
    -------
    out : ndarray
        1-D array of the polynomial's coefficients If all the roots are
        real, then `out` is also real, otherwise it is complex.  (see
        Examples below).

    See Also
    --------
    numpy.polynomial.chebyshev.chebfromroots
    numpy.polynomial.legendre.legfromroots
    numpy.polynomial.laguerre.lagfromroots
    numpy.polynomial.hermite.hermfromroots
    numpy.polynomial.hermite_e.hermefromroots

    Notes
    -----
    The coefficients are determined by multiplying together linear factors
    of the form ``(x - r_i)``, i.e.

    .. math:: p(x) = (x - r_0) (x - r_1) ... (x - r_n)

    where ``n == len(roots) - 1``; note that this implies that ``1`` is always
    returned for :math:`a_n`.

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> P.polyfromroots((-1,0,1))  # x(x - 1)(x + 1) = x^3 - x
    array([ 0., -1.,  0.,  1.])
    >>> j = complex(0,1)
    >>> P.polyfromroots((-j,j))  # complex returned, though values are real
    array([1.+0.j,  0.+0.j,  1.+0.j])

    """
    return pu._fromroots(polyline, polymul, roots)


def polyadd(c1, c2):
    """
    Add one polynomial to another.

    Returns the sum of two polynomials `c1` + `c2`.  The arguments are
    sequences of coefficients from lowest order term to highest, i.e.,
    [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of polynomial coefficients ordered from low to high.

    Returns
    -------
    out : ndarray
        The coefficient array representing their sum.

    See Also
    --------
    polysub, polymulx, polymul, polydiv, polypow

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c1 = (1, 2, 3)
    >>> c2 = (3, 2, 1)
    >>> sum = P.polyadd(c1,c2); sum
    array([4.,  4.,  4.])
    >>> P.polyval(2, sum)  # 4 + 4(2) + 4(2**2)
    28.0

    """
    return pu._add(c1, c2)


def polysub(c1, c2):
    """
    Subtract one polynomial from another.

    Returns the difference of two polynomials `c1` - `c2`.  The arguments
    are sequences of coefficients from lowest order term to highest, i.e.,
    [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of polynomial coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of coefficients representing their difference.

    See Also
    --------
    polyadd, polymulx, polymul, polydiv, polypow

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c1 = (1, 2, 3)
    >>> c2 = (3, 2, 1)
    >>> P.polysub(c1,c2)
    array([-2.,  0.,  2.])
    >>> P.polysub(c2, c1)  # -P.polysub(c1,c2)
    array([ 2.,  0., -2.])

    """
    return pu._sub(c1, c2)


def polymulx(c):
    """Multiply a polynomial by x.

    Multiply the polynomial `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-D array of polynomial coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    See Also
    --------
    polyadd, polysub, polymul, polydiv, polypow

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = (1, 2, 3)
    >>> P.polymulx(c)
    array([0., 1., 2., 3.])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    # The zero series needs special treatment
    if len(c) == 1 and c[0] == 0:
        return c

    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0] * 0
    prd[1:] = c
    return prd


def polymul(c1, c2):
    """
    Multiply one polynomial by another.

    Returns the product of two polynomials `c1` * `c2`.  The arguments are
    sequences of coefficients, from lowest order term to highest, e.g.,
    [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2.``

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of coefficients representing a polynomial, relative to the
        "standard" basis, and ordered from lowest order term to highest.

    Returns
    -------
    out : ndarray
        Of the coefficients of their product.

    See Also
    --------
    polyadd, polysub, polymulx, polydiv, polypow

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c1 = (1, 2, 3)
    >>> c2 = (3, 2, 1)
    >>> P.polymul(c1, c2)
    array([  3.,   8.,  14.,   8.,   3.])

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    ret = np.convolve(c1, c2)
    return pu.trimseq(ret)


def polydiv(c1, c2):
    """
    Divide one polynomial by another.

    Returns the quotient-with-remainder of two polynomials `c1` / `c2`.
    The arguments are sequences of coefficients, from lowest order term
    to highest, e.g., [1,2,3] represents ``1 + 2*x + 3*x**2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of polynomial coefficients ordered from low to high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of coefficient series representing the quotient and remainder.

    See Also
    --------
    polyadd, polysub, polymulx, polymul, polypow

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c1 = (1, 2, 3)
    >>> c2 = (3, 2, 1)
    >>> P.polydiv(c1, c2)
    (array([3.]), array([-8., -4.]))
    >>> P.polydiv(c2, c1)
    (array([ 0.33333333]), array([ 2.66666667,  1.33333333]))  # may vary

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    if c2[-1] == 0:
        raise ZeroDivisionError  # FIXME: add message with details to exception

    # note: this is more efficient than `pu._div(polymul, c1, c2)`
    lc1 = len(c1)
    lc2 = len(c2)
    if lc1 < lc2:
        return c1[:1] * 0, c1
    elif lc2 == 1:
        return c1 / c2[-1], c1[:1] * 0
    else:
        dlen = lc1 - lc2
        scl = c2[-1]
        c2 = c2[:-1] / scl
        i = dlen
        j = lc1 - 1
        while i >= 0:
            c1[i:j] -= c2 * c1[j]
            i -= 1
            j -= 1
        return c1[j + 1:] / scl, pu.trimseq(c1[:j + 1])


def polypow(c, pow, maxpower=None):
    """Raise a polynomial to a power.

    Returns the polynomial `c` raised to the power `pow`. The argument
    `c` is a sequence of coefficients ordered from low to high. i.e.,
    [1,2,3] is the series  ``1 + 2*x + 3*x**2.``

    Parameters
    ----------
    c : array_like
        1-D array of array of series coefficients ordered from low to
        high degree.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Power series of power.

    See Also
    --------
    polyadd, polysub, polymulx, polymul, polydiv

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> P.polypow([1, 2, 3], 2)
    array([ 1., 4., 10., 12., 9.])

    """
    # note: this is more efficient than `pu._pow(polymul, c1, c2)`, as it
    # avoids calling `as_series` repeatedly
    return pu._pow(np.convolve, c, pow, maxpower)


def polyder(c, m=1, scl=1, axis=0):
    """
    Differentiate a polynomial.

    Returns the polynomial coefficients `c` differentiated `m` times along
    `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable).  The
    argument `c` is an array of coefficients from low to high degree along
    each axis, e.g., [1,2,3] represents the polynomial ``1 + 2*x + 3*x**2``
    while [[1,2],[1,2]] represents ``1 + 1*x + 2*y + 2*x*y`` if axis=0 is
    ``x`` and axis=1 is ``y``.

    Parameters
    ----------
    c : array_like
        Array of polynomial coefficients. If c is multidimensional the
        different axis correspond to different variables with the degree
        in each axis given by the corresponding index.
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change
        of variable. (Default: 1)
    axis : int, optional
        Axis over which the derivative is taken. (Default: 0).

    Returns
    -------
    der : ndarray
        Polynomial coefficients of the derivative.

    See Also
    --------
    polyint

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = (1, 2, 3, 4)
    >>> P.polyder(c)  # (d/dx)(c)
    array([  2.,   6.,  12.])
    >>> P.polyder(c, 3)  # (d**3/dx**3)(c)
    array([24.])
    >>> P.polyder(c, scl=-1)  # (d/d(-x))(c)
    array([ -2.,  -6., -12.])
    >>> P.polyder(c, 2, -1)  # (d**2/d(-x)**2)(c)
    array([  6.,  24.])

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        # astype fails with NA
        c = c + 0.0
    cdt = c.dtype
    cnt = pu._as_int(m, "the order of derivation")
    iaxis = pu._as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = np.moveaxis(c, iaxis, 0)
    n = len(c)
    if cnt >= n:
        c = c[:1] * 0
    else:
        for i in range(cnt):
            n = n - 1
            c *= scl
            der = np.empty((n,) + c.shape[1:], dtype=cdt)
            for j in range(n, 0, -1):
                der[j - 1] = j * c[j]
            c = der
    c = np.moveaxis(c, 0, iaxis)
    return c


def polyint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a polynomial.

    Returns the polynomial coefficients `c` integrated `m` times from
    `lbnd` along `axis`.  At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.) The argument `c` is an array of
    coefficients, from low to high degree along each axis, e.g., [1,2,3]
    represents the polynomial ``1 + 2*x + 3*x**2`` while [[1,2],[1,2]]
    represents ``1 + 1*x + 2*y + 2*x*y`` if axis=0 is ``x`` and axis=1 is
    ``y``.

    Parameters
    ----------
    c : array_like
        1-D array of polynomial coefficients, ordered from low to high.
    m : int, optional
        Order of integration, must be positive. (Default: 1)
    k : {[], list, scalar}, optional
        Integration constant(s).  The value of the first integral at zero
        is the first value in the list, the value of the second integral
        at zero is the second value, etc.  If ``k == []`` (the default),
        all constants are set to zero.  If ``m == 1``, a single scalar can
        be given instead of a list.
    lbnd : scalar, optional
        The lower bound of the integral. (Default: 0)
    scl : scalar, optional
        Following each integration the result is *multiplied* by `scl`
        before the integration constant is added. (Default: 1)
    axis : int, optional
        Axis over which the integral is taken. (Default: 0).

    Returns
    -------
    S : ndarray
        Coefficient array of the integral.

    Raises
    ------
    ValueError
        If ``m < 1``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    polyder

    Notes
    -----
    Note that the result of each integration is *multiplied* by `scl`.  Why
    is this important to note?  Say one is making a linear change of
    variable :math:`u = ax + b` in an integral relative to `x`. Then
    :math:`dx = du/a`, so one will need to set `scl` equal to
    :math:`1/a` - perhaps not what one would have first thought.

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = (1, 2, 3)
    >>> P.polyint(c)  # should return array([0, 1, 1, 1])
    array([0.,  1.,  1.,  1.])
    >>> P.polyint(c, 3)  # should return array([0, 0, 0, 1/6, 1/12, 1/20])
     array([ 0.        ,  0.        ,  0.        ,  0.16666667,  0.08333333, # may vary
             0.05      ])
    >>> P.polyint(c, k=3)  # should return array([3, 1, 1, 1])
    array([3.,  1.,  1.,  1.])
    >>> P.polyint(c,lbnd=-2)  # should return array([6, 1, 1, 1])
    array([6.,  1.,  1.,  1.])
    >>> P.polyint(c,scl=-2)  # should return array([0, -2, -2, -2])
    array([ 0., -2., -2., -2.])

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        # astype doesn't preserve mask attribute.
        c = c + 0.0
    cdt = c.dtype
    if not np.iterable(k):
        k = [k]
    cnt = pu._as_int(m, "the order of integration")
    iaxis = pu._as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if np.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if np.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    k = list(k) + [0] * (cnt - len(k))
    c = np.moveaxis(c, iaxis, 0)
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and np.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=cdt)
            tmp[0] = c[0] * 0
            tmp[1] = c[0]
            for j in range(1, n):
                tmp[j + 1] = c[j] / (j + 1)
            tmp[0] += k[i] - polyval(lbnd, tmp)
            c = tmp
    c = np.moveaxis(c, 0, iaxis)
    return c


def polyval(x, c, tensor=True):
    """
    Evaluate a polynomial at points x.

    If `c` is of length ``n + 1``, this function returns the value

    .. math:: p(x) = c_0 + c_1 * x + ... + c_n * x^n

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.

    If `c` is a 1-D array, then ``p(x)`` will have the same shape as `x`.  If
    `c` is multidimensional, then the shape of the result depends on the
    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
    scalars have shape (,).

    Trailing zeros in the coefficients will be used in the evaluation, so
    they should be avoided if efficiency is a concern.

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        with themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimensional the
        remaining indices enumerate multiple polynomials. In the two
        dimensional case the coefficients may be thought of as stored in
        the columns of `c`.
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.

    Returns
    -------
    values : ndarray, compatible object
        The shape of the returned array is described above.

    See Also
    --------
    polyval2d, polygrid2d, polyval3d, polygrid3d

    Notes
    -----
    The evaluation uses Horner's method.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial.polynomial import polyval
    >>> polyval(1, [1,2,3])
    6.0
    >>> a = np.arange(4).reshape(2,2)
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> polyval(a, [1, 2, 3])
    array([[ 1.,   6.],
           [17.,  34.]])
    >>> coef = np.arange(4).reshape(2, 2)  # multidimensional coefficients
    >>> coef
    array([[0, 1],
           [2, 3]])
    >>> polyval([1, 2], coef, tensor=True)
    array([[2.,  4.],
           [4.,  7.]])
    >>> polyval([1, 2], coef, tensor=False)
    array([2.,  7.])

    """
    c = np.array(c, ndmin=1, copy=None)
    if c.dtype.char in '?bBhHiIlLqQpP':
        # astype fails with NA
        c = c + 0.0
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    c0 = c[-1] + x * 0
    for i in range(2, len(c) + 1):
        c0 = c[-i] + c0 * x
    return c0


def polyvalfromroots(x, r, tensor=True):
    """
    Evaluate a polynomial specified by its roots at points x.

    If `r` is of length ``N``, this function returns the value

    .. math:: p(x) = \\prod_{n=1}^{N} (x - r_n)

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `r`.

    If `r` is a 1-D array, then ``p(x)`` will have the same shape as `x`.  If `r`
    is multidimensional, then the shape of the result depends on the value of
    `tensor`. If `tensor` is ``True`` the shape will be r.shape[1:] + x.shape;
    that is, each polynomial is evaluated at every value of `x`. If `tensor` is
    ``False``, the shape will be r.shape[1:]; that is, each polynomial is
    evaluated only for the corresponding broadcast value of `x`. Note that
    scalars have shape (,).

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        with themselves and with the elements of `r`.
    r : array_like
        Array of roots. If `r` is multidimensional the first index is the
        root index, while the remaining indices enumerate multiple
        polynomials. For instance, in the two dimensional case the roots
        of each polynomial may be thought of as stored in the columns of `r`.
    tensor : boolean, optional
        If True, the shape of the roots array is extended with ones on the
        right, one for each dimension of `x`. Scalars have dimension 0 for this
        action. The result is that every column of coefficients in `r` is
        evaluated for every element of `x`. If False, `x` is broadcast over the
        columns of `r` for the evaluation.  This keyword is useful when `r` is
        multidimensional. The default value is True.

    Returns
    -------
    values : ndarray, compatible object
        The shape of the returned array is described above.

    See Also
    --------
    polyroots, polyfromroots, polyval

    Examples
    --------
    >>> from numpy.polynomial.polynomial import polyvalfromroots
    >>> polyvalfromroots(1, [1, 2, 3])
    0.0
    >>> a = np.arange(4).reshape(2, 2)
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> polyvalfromroots(a, [-1, 0, 1])
    array([[-0.,   0.],
           [ 6.,  24.]])
    >>> r = np.arange(-2, 2).reshape(2,2)  # multidimensional coefficients
    >>> r # each column of r defines one polynomial
    array([[-2, -1],
           [ 0,  1]])
    >>> b = [-2, 1]
    >>> polyvalfromroots(b, r, tensor=True)
    array([[-0.,  3.],
           [ 3., 0.]])
    >>> polyvalfromroots(b, r, tensor=False)
    array([-0.,  0.])

    """
    r = np.array(r, ndmin=1, copy=None)
    if r.dtype.char in '?bBhHiIlLqQpP':
        r = r.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray):
        if tensor:
            r = r.reshape(r.shape + (1,) * x.ndim)
        elif x.ndim >= r.ndim:
            raise ValueError("x.ndim must be < r.ndim when tensor == False")
    return np.prod(x - r, axis=0)


def polyval2d(x, y, c):
    """
    Evaluate a 2-D polynomial at points (x, y).

    This function returns the value

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * x^i * y^j

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars and they
    must have the same shape after conversion. In either case, either `x`
    and `y` or their elements must support multiplication and addition both
    with themselves and with the elements of `c`.

    If `c` has fewer than two dimensions, ones are implicitly appended to
    its shape to make it 2-D. The shape of the result will be c.shape[2:] +
    x.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points ``(x, y)``,
        where `x` and `y` must have the same shape. If `x` or `y` is a list
        or tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term
        of multi-degree i,j is contained in ``c[i,j]``. If `c` has
        dimension greater than two the remaining indices enumerate multiple
        sets of coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points formed with
        pairs of corresponding values from `x` and `y`.

    See Also
    --------
    polyval, polygrid2d, polyval3d, polygrid3d

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = ((1, 2, 3), (4, 5, 6))
    >>> P.polyval2d(1, 1, c)
    21.0

    """
    return pu._valnd(polyval, c, x, y)


def polygrid2d(x, y, c):
    """
    Evaluate a 2-D polynomial on the Cartesian product of x and y.

    This function returns the values:

    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * a^i * b^j

    where the points ``(a, b)`` consist of all pairs formed by taking
    `a` from `x` and `b` from `y`. The resulting points form a grid with
    `x` in the first dimension and `y` in the second.

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars. In either
    case, either `x` and `y` or their elements must support multiplication
    and addition both with themselves and with the elements of `c`.

    If `c` has fewer than two dimensions, ones are implicitly appended to
    its shape to make it 2-D. The shape of the result will be c.shape[2:] +
    x.shape + y.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points in the
        Cartesian product of `x` and `y`.  If `x` or `y` is a list or
        tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    polyval, polyval2d, polyval3d, polygrid3d

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = ((1, 2, 3), (4, 5, 6))
    >>> P.polygrid2d([0, 1], [0, 1], c)
    array([[ 1.,  6.],
           [ 5., 21.]])

    """
    return pu._gridnd(polyval, c, x, y)


def polyval3d(x, y, z, c):
    """
    Evaluate a 3-D polynomial at points (x, y, z).

    This function returns the values:

    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * x^i * y^j * z^k

    The parameters `x`, `y`, and `z` are converted to arrays only if
    they are tuples or a lists, otherwise they are treated as a scalars and
    they must have the same shape after conversion. In either case, either
    `x`, `y`, and `z` or their elements must support multiplication and
    addition both with themselves and with the elements of `c`.

    If `c` has fewer than 3 dimensions, ones are implicitly appended to its
    shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible object
        The three dimensional series is evaluated at the points
        ``(x, y, z)``, where `x`, `y`, and `z` must have the same shape.  If
        any of `x`, `y`, or `z` is a list or tuple, it is first converted
        to an ndarray, otherwise it is left unchanged and if it isn't an
        ndarray it is  treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
        greater than 3 the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the multidimensional polynomial on points formed with
        triples of corresponding values from `x`, `y`, and `z`.

    See Also
    --------
    polyval, polyval2d, polygrid2d, polygrid3d

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    >>> P.polyval3d(1, 1, 1, c)
    45.0

    """
    return pu._valnd(polyval, c, x, y, z)


def polygrid3d(x, y, z, c):
    """
    Evaluate a 3-D polynomial on the Cartesian product of x, y and z.

    This function returns the values:

    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * a^i * b^j * c^k

    where the points ``(a, b, c)`` consist of all triples formed by taking
    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
    a grid with `x` in the first dimension, `y` in the second, and `z` in
    the third.

    The parameters `x`, `y`, and `z` are converted to arrays only if they
    are tuples or a lists, otherwise they are treated as a scalars. In
    either case, either `x`, `y`, and `z` or their elements must support
    multiplication and addition both with themselves and with the elements
    of `c`.

    If `c` has fewer than three dimensions, ones are implicitly appended to
    its shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape + y.shape + z.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible objects
        The three dimensional series is evaluated at the points in the
        Cartesian product of `x`, `y`, and `z`.  If `x`, `y`, or `z` is a
        list or tuple, it is first converted to an ndarray, otherwise it is
        left unchanged and, if it isn't an ndarray, it is treated as a
        scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    polyval, polyval2d, polygrid2d, polyval3d

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    >>> P.polygrid3d([0, 1], [0, 1], [0, 1], c)
    array([[ 1., 13.],
           [ 6., 51.]])

    """
    return pu._gridnd(polyval, c, x, y, z)


def polyvander(x, deg):
    """Vandermonde matrix of given degree.

    Returns the Vandermonde matrix of degree `deg` and sample points
    `x`. The Vandermonde matrix is defined by

    .. math:: V[..., i] = x^i,

    where ``0 <= i <= deg``. The leading indices of `V` index the elements of
    `x` and the last index is the power of `x`.

    If `c` is a 1-D array of coefficients of length ``n + 1`` and `V` is the
    matrix ``V = polyvander(x, n)``, then ``np.dot(V, c)`` and
    ``polyval(x, c)`` are the same up to roundoff. This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of polynomials of the same degree and sample points.

    Parameters
    ----------
    x : array_like
        Array of points. The dtype is converted to float64 or complex128
        depending on whether any of the elements are complex. If `x` is
        scalar it is converted to a 1-D array.
    deg : int
        Degree of the resulting matrix.

    Returns
    -------
    vander : ndarray.
        The Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where the last index is the power of `x`.
        The dtype will be the same as the converted `x`.

    See Also
    --------
    polyvander2d, polyvander3d

    Examples
    --------
    The Vandermonde matrix of degree ``deg = 5`` and sample points
    ``x = [-1, 2, 3]`` contains the element-wise powers of `x`
    from 0 to 5 as its columns.

    >>> from numpy.polynomial import polynomial as P
    >>> x, deg = [-1, 2, 3], 5
    >>> P.polyvander(x=x, deg=deg)
    array([[  1.,  -1.,   1.,  -1.,   1.,  -1.],
           [  1.,   2.,   4.,   8.,  16.,  32.],
           [  1.,   3.,   9.,  27.,  81., 243.]])

    """
    ideg = pu._as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.array(x, copy=None, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = np.empty(dims, dtype=dtyp)
    v[0] = x * 0 + 1
    if ideg > 0:
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = v[i - 1] * x
    return np.moveaxis(v, 0, -1)


def polyvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y)``. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = x^i * y^j,

    where ``0 <= i <= deg[0]`` and ``0 <= j <= deg[1]``. The leading indices of
    `V` index the points ``(x, y)`` and the last index encodes the powers of
    `x` and `y`.

    If ``V = polyvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``polyval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D polynomials
    of the same degrees and sample points.

    Parameters
    ----------
    x, y : array_like
        Arrays of point coordinates, all of the same shape. The dtypes
        will be converted to either float64 or complex128 depending on
        whether any of the elements are complex. Scalars are converted to
        1-D arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg].

    Returns
    -------
    vander2d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same
        as the converted `x` and `y`.

    See Also
    --------
    polyvander, polyvander3d, polyval2d, polyval3d

    Examples
    --------
    >>> import numpy as np

    The 2-D pseudo-Vandermonde matrix of degree ``[1, 2]`` and sample
    points ``x = [-1, 2]`` and ``y = [1, 3]`` is as follows:

    >>> from numpy.polynomial import polynomial as P
    >>> x = np.array([-1, 2])
    >>> y = np.array([1, 3])
    >>> m, n = 1, 2
    >>> deg = np.array([m, n])
    >>> V = P.polyvander2d(x=x, y=y, deg=deg)
    >>> V
    array([[ 1.,  1.,  1., -1., -1., -1.],
           [ 1.,  3.,  9.,  2.,  6., 18.]])

    We can verify the columns for any ``0 <= i <= m`` and ``0 <= j <= n``:

    >>> i, j = 0, 1
    >>> V[:, (deg[1]+1)*i + j] == x**i * y**j
    array([ True,  True])

    The (1D) Vandermonde matrix of sample points ``x`` and degree ``m`` is a
    special case of the (2D) pseudo-Vandermonde matrix with ``y`` points all
    zero and degree ``[m, 0]``.

    >>> P.polyvander2d(x=x, y=0*x, deg=(m, 0)) == P.polyvander(x=x, deg=m)
    array([[ True,  True],
           [ True,  True]])

    """
    return pu._vander_nd_flat((polyvander, polyvander), (x, y), deg)


def polyvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y, z)``. If `l`, `m`, `n` are the given degrees in `x`, `y`, `z`,
    then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = x^i * y^j * z^k,

    where ``0 <= i <= l``, ``0 <= j <= m``, and ``0 <= j <= n``.  The leading
    indices of `V` index the points ``(x, y, z)`` and the last index encodes
    the powers of `x`, `y`, and `z`.

    If ``V = polyvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and  ``np.dot(V, c.flat)`` and ``polyval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D polynomials
    of the same degrees and sample points.

    Parameters
    ----------
    x, y, z : array_like
        Arrays of point coordinates, all of the same shape. The dtypes will
        be converted to either float64 or complex128 depending on whether
        any of the elements are complex. Scalars are converted to 1-D
        arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg, z_deg].

    Returns
    -------
    vander3d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will
        be the same as the converted `x`, `y`, and `z`.

    See Also
    --------
    polyvander, polyvander3d, polyval2d, polyval3d

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial import polynomial as P
    >>> x = np.asarray([-1, 2, 1])
    >>> y = np.asarray([1, -2, -3])
    >>> z = np.asarray([2, 2, 5])
    >>> l, m, n = [2, 2, 1]
    >>> deg = [l, m, n]
    >>> V = P.polyvander3d(x=x, y=y, z=z, deg=deg)
    >>> V
    array([[  1.,   2.,   1.,   2.,   1.,   2.,  -1.,  -2.,  -1.,
             -2.,  -1.,  -2.,   1.,   2.,   1.,   2.,   1.,   2.],
           [  1.,   2.,  -2.,  -4.,   4.,   8.,   2.,   4.,  -4.,
             -8.,   8.,  16.,   4.,   8.,  -8., -16.,  16.,  32.],
           [  1.,   5.,  -3., -15.,   9.,  45.,   1.,   5.,  -3.,
            -15.,   9.,  45.,   1.,   5.,  -3., -15.,   9.,  45.]])

    We can verify the columns for any ``0 <= i <= l``, ``0 <= j <= m``,
    and ``0 <= k <= n``

    >>> i, j, k = 2, 1, 0
    >>> V[:, (m+1)*(n+1)*i + (n+1)*j + k] == x**i * y**j * z**k
    array([ True,  True,  True])

    """
    return pu._vander_nd_flat((polyvander, polyvander, polyvander), (x, y, z), deg)


def polyfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least-squares fit of a polynomial to data.

    Return the coefficients of a polynomial of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * x + ... + c_n * x^n,

    where `n` is `deg`.

    Parameters
    ----------
    x : array_like, shape (`M`,)
        x-coordinates of the `M` sample (data) points ``(x[i], y[i])``.
    y : array_like, shape (`M`,) or (`M`, `K`)
        y-coordinates of the sample points.  Several sets of sample points
        sharing the same x-coordinates can be (independently) fit with one
        call to `polyfit` by passing in for `y` a 2-D array that contains
        one data set per column.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If `deg` is a single integer
        all terms up to and including the `deg`'th term are included in the
        fit. For NumPy versions >= 1.11.0 a list of integers specifying the
        degrees of the terms to include may be used instead.
    rcond : float, optional
        Relative condition number of the fit.  Singular values smaller
        than `rcond`, relative to the largest singular value, will be
        ignored.  The default value is ``len(x)*eps``, where `eps` is the
        relative precision of the platform's float type, about 2e-16 in
        most cases.
    full : bool, optional
        Switch determining the nature of the return value.  When ``False``
        (the default) just the coefficients are returned; when ``True``,
        diagnostic information from the singular value decomposition (used
        to solve the fit's matrix equation) is also returned.
    w : array_like, shape (`M`,), optional
        Weights. If not None, the weight ``w[i]`` applies to the unsquared
        residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
        chosen so that the errors of the products ``w[i]*y[i]`` all have the
        same variance.  When using inverse-variance weighting, use
        ``w[i] = 1/sigma(y[i])``.  The default value is None.

    Returns
    -------
    coef : ndarray, shape (`deg` + 1,) or (`deg` + 1, `K`)
        Polynomial coefficients ordered from low to high.  If `y` was 2-D,
        the coefficients in column `k` of `coef` represent the polynomial
        fit to the data in `y`'s `k`-th column.

    [residuals, rank, singular_values, rcond] : list
        These values are only returned if ``full == True``

        - residuals -- sum of squared residuals of the least squares fit
        - rank -- the numerical rank of the scaled Vandermonde matrix
        - singular_values -- singular values of the scaled Vandermonde matrix
        - rcond -- value of `rcond`.

        For more details, see `numpy.linalg.lstsq`.

    Raises
    ------
    RankWarning
        Raised if the matrix in the least-squares fit is rank deficient.
        The warning is only raised if ``full == False``.  The warnings can
        be turned off by:

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    numpy.polynomial.chebyshev.chebfit
    numpy.polynomial.legendre.legfit
    numpy.polynomial.laguerre.lagfit
    numpy.polynomial.hermite.hermfit
    numpy.polynomial.hermite_e.hermefit
    polyval : Evaluates a polynomial.
    polyvander : Vandermonde matrix for powers.
    numpy.linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the polynomial `p` that minimizes
    the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where the :math:`w_j` are the weights. This problem is solved by
    setting up the (typically) over-determined matrix equation:

    .. math:: V(x) * c = w * y,

    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
    coefficients to be solved for, `w` are the weights, and `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of `V`.

    If some of the singular values of `V` are so small that they are
    neglected (and `full` == ``False``), a `~exceptions.RankWarning` will be
    raised.  This means that the coefficient values may be poorly determined.
    Fitting to a lower order polynomial will usually get rid of the warning
    (but may not be what you want, of course; if you have independent
    reason(s) for choosing the degree which isn't working, you may have to:
    a) reconsider those reasons, and/or b) reconsider the quality of your
    data).  The `rcond` parameter can also be set to a value smaller than
    its default, but the resulting fit may be spurious and have large
    contributions from roundoff error.

    Polynomial fits using double precision tend to "fail" at about
    (polynomial) degree 20. Fits using Chebyshev or Legendre series are
    generally better conditioned, but much can still depend on the
    distribution of the sample points and the smoothness of the data.  If
    the quality of the fit is inadequate, splines may be a good
    alternative.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial import polynomial as P
    >>> x = np.linspace(-1,1,51)  # x "data": [-1, -0.96, ..., 0.96, 1]
    >>> rng = np.random.default_rng()
    >>> err = rng.normal(size=len(x))
    >>> y = x**3 - x + err  # x^3 - x + Gaussian noise
    >>> c, stats = P.polyfit(x,y,3,full=True)
    >>> c # c[0], c[1] approx. -1, c[2] should be approx. 0, c[3] approx. 1
    array([ 0.23111996, -1.02785049, -0.2241444 ,  1.08405657]) # may vary
    >>> stats # note the large SSR, explaining the rather poor results
    [array([48.312088]),                                        # may vary
     4,
     array([1.38446749, 1.32119158, 0.50443316, 0.28853036]),
     1.1324274851176597e-14]

    Same thing without the added noise

    >>> y = x**3 - x
    >>> c, stats = P.polyfit(x,y,3,full=True)
    >>> c # c[0], c[1] ~= -1, c[2] should be "very close to 0", c[3] ~= 1
    array([-6.73496154e-17, -1.00000000e+00,  0.00000000e+00,  1.00000000e+00])
    >>> stats # note the minuscule SSR
    [array([8.79579319e-31]),
     np.int32(4),
     array([1.38446749, 1.32119158, 0.50443316, 0.28853036]),
     1.1324274851176597e-14]

    """
    return pu._fit(polyvander, x, y, deg, rcond, full, w)


def polycompanion(c):
    """
    Return the companion matrix of c.

    The companion matrix for power series cannot be made symmetric by
    scaling the basis, so this function differs from those for the
    orthogonal polynomials.

    Parameters
    ----------
    c : array_like
        1-D array of polynomial coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Companion matrix of dimensions (deg, deg).

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> c = (1, 2, 3)
    >>> P.polycompanion(c)
    array([[ 0.        , -0.33333333],
           [ 1.        , -0.66666667]])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    if len(c) == 2:
        return np.array([[-c[0] / c[1]]])

    n = len(c) - 1
    mat = np.zeros((n, n), dtype=c.dtype)
    bot = mat.reshape(-1)[n::n + 1]
    bot[...] = 1
    mat[:, -1] -= c[:-1] / c[-1]
    return mat


def polyroots(c):
    """
    Compute the roots of a polynomial.

    Return the roots (a.k.a. "zeros") of the polynomial

    .. math:: p(x) = \\sum_i c[i] * x^i.

    Parameters
    ----------
    c : 1-D array_like
        1-D array of polynomial coefficients.

    Returns
    -------
    out : ndarray
        Array of the roots of the polynomial. If all the roots are real,
        then `out` is also real, otherwise it is complex.

    See Also
    --------
    numpy.polynomial.chebyshev.chebroots
    numpy.polynomial.legendre.legroots
    numpy.polynomial.laguerre.lagroots
    numpy.polynomial.hermite.hermroots
    numpy.polynomial.hermite_e.hermeroots

    Notes
    -----
    The root estimates are obtained as the eigenvalues of the companion
    matrix, Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the power series for such
    values. Roots with multiplicity greater than 1 will also show larger
    errors as the value of the series near such points is relatively
    insensitive to errors in the roots. Isolated roots near the origin can
    be improved by a few iterations of Newton's method.

    Examples
    --------
    >>> import numpy.polynomial.polynomial as poly
    >>> poly.polyroots(poly.polyfromroots((-1,0,1)))
    array([-1.,  0.,  1.])
    >>> poly.polyroots(poly.polyfromroots((-1,0,1))).dtype
    dtype('float64')
    >>> j = complex(0,1)
    >>> poly.polyroots(poly.polyfromroots((-j,0,j)))
    array([  0.00000000e+00+0.j,   0.00000000e+00+1.j,   2.77555756e-17-1.j])  # may vary

    """  # noqa: E501
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) < 2:
        return np.array([], dtype=c.dtype)
    if len(c) == 2:
        return np.array([-c[0] / c[1]])

    m = polycompanion(c)
    r = la.eigvals(m)
    r.sort()
    return r


#
# polynomial class
#

class Polynomial(ABCPolyBase):
    """A power series class.

    The Polynomial class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    coef : array_like
        Polynomial coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` give ``1 + 2*x + 3*x**2``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is [-1., 1.].
    window : (2,) array_like, optional
        Window, see `domain` for its use. The default value is [-1., 1.].
    symbol : str, optional
        Symbol used to represent the independent variable in string
        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    """
    # Virtual Functions
    _add = staticmethod(polyadd)
    _sub = staticmethod(polysub)
    _mul = staticmethod(polymul)
    _div = staticmethod(polydiv)
    _pow = staticmethod(polypow)
    _val = staticmethod(polyval)
    _int = staticmethod(polyint)
    _der = staticmethod(polyder)
    _fit = staticmethod(polyfit)
    _line = staticmethod(polyline)
    _roots = staticmethod(polyroots)
    _fromroots = staticmethod(polyfromroots)

    # Virtual properties
    domain = np.array(polydomain)
    window = np.array(polydomain)
    basis_name = None

    @classmethod
    def _str_term_unicode(cls, i, arg_str):
        if i == '1':
            return f"·{arg_str}"
        else:
            return f"·{arg_str}{i.translate(cls._superscript_mapping)}"

    @staticmethod
    def _str_term_ascii(i, arg_str):
        if i == '1':
            return f" {arg_str}"
        else:
            return f" {arg_str}**{i}"

    @staticmethod
    def _repr_latex_term(i, arg_str, needs_parens):
        if needs_parens:
            arg_str = rf"\left({arg_str}\right)"
        if i == 0:
            return '1'
        elif i == 1:
            return arg_str
        else:
            return f"{arg_str}^{{{i}}}"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\assumptions_generated.py ===
"""
Do NOT manually edit this file.
Instead, run ./bin/ask_update.py.
"""

defined_facts = [
    'algebraic',
    'antihermitian',
    'commutative',
    'complex',
    'composite',
    'even',
    'extended_negative',
    'extended_nonnegative',
    'extended_nonpositive',
    'extended_nonzero',
    'extended_positive',
    'extended_real',
    'finite',
    'hermitian',
    'imaginary',
    'infinite',
    'integer',
    'irrational',
    'negative',
    'noninteger',
    'nonnegative',
    'nonpositive',
    'nonzero',
    'odd',
    'positive',
    'prime',
    'rational',
    'real',
    'transcendental',
    'zero',
] # defined_facts


full_implications = dict( [
    # Implications of algebraic = True:
    (('algebraic', True), set( (
        ('commutative', True),
        ('complex', True),
        ('finite', True),
        ('infinite', False),
        ('transcendental', False),
       ) ),
     ),
    # Implications of algebraic = False:
    (('algebraic', False), set( (
        ('composite', False),
        ('even', False),
        ('integer', False),
        ('odd', False),
        ('prime', False),
        ('rational', False),
        ('zero', False),
       ) ),
     ),
    # Implications of antihermitian = True:
    (('antihermitian', True), set( (
       ) ),
     ),
    # Implications of antihermitian = False:
    (('antihermitian', False), set( (
        ('imaginary', False),
       ) ),
     ),
    # Implications of commutative = True:
    (('commutative', True), set( (
       ) ),
     ),
    # Implications of commutative = False:
    (('commutative', False), set( (
        ('algebraic', False),
        ('complex', False),
        ('composite', False),
        ('even', False),
        ('extended_negative', False),
        ('extended_nonnegative', False),
        ('extended_nonpositive', False),
        ('extended_nonzero', False),
        ('extended_positive', False),
        ('extended_real', False),
        ('imaginary', False),
        ('integer', False),
        ('irrational', False),
        ('negative', False),
        ('noninteger', False),
        ('nonnegative', False),
        ('nonpositive', False),
        ('nonzero', False),
        ('odd', False),
        ('positive', False),
        ('prime', False),
        ('rational', False),
        ('real', False),
        ('transcendental', False),
        ('zero', False),
       ) ),
     ),
    # Implications of complex = True:
    (('complex', True), set( (
        ('commutative', True),
        ('finite', True),
        ('infinite', False),
       ) ),
     ),
    # Implications of complex = False:
    (('complex', False), set( (
        ('algebraic', False),
        ('composite', False),
        ('even', False),
        ('imaginary', False),
        ('integer', False),
        ('irrational', False),
        ('negative', False),
        ('nonnegative', False),
        ('nonpositive', False),
        ('nonzero', False),
        ('odd', False),
        ('positive', False),
        ('prime', False),
        ('rational', False),
        ('real', False),
        ('transcendental', False),
        ('zero', False),
       ) ),
     ),
    # Implications of composite = True:
    (('composite', True), set( (
        ('algebraic', True),
        ('commutative', True),
        ('complex', True),
        ('extended_negative', False),
        ('extended_nonnegative', True),
        ('extended_nonpositive', False),
        ('extended_nonzero', True),
        ('extended_positive', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('integer', True),
        ('irrational', False),
        ('negative', False),
        ('noninteger', False),
        ('nonnegative', True),
        ('nonpositive', False),
        ('nonzero', True),
        ('positive', True),
        ('prime', False),
        ('rational', True),
        ('real', True),
        ('transcendental', False),
        ('zero', False),
       ) ),
     ),
    # Implications of composite = False:
    (('composite', False), set( (
       ) ),
     ),
    # Implications of even = True:
    (('even', True), set( (
        ('algebraic', True),
        ('commutative', True),
        ('complex', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('integer', True),
        ('irrational', False),
        ('noninteger', False),
        ('odd', False),
        ('rational', True),
        ('real', True),
        ('transcendental', False),
       ) ),
     ),
    # Implications of even = False:
    (('even', False), set( (
        ('zero', False),
       ) ),
     ),
    # Implications of extended_negative = True:
    (('extended_negative', True), set( (
        ('commutative', True),
        ('composite', False),
        ('extended_nonnegative', False),
        ('extended_nonpositive', True),
        ('extended_nonzero', True),
        ('extended_positive', False),
        ('extended_real', True),
        ('imaginary', False),
        ('nonnegative', False),
        ('positive', False),
        ('prime', False),
        ('zero', False),
       ) ),
     ),
    # Implications of extended_negative = False:
    (('extended_negative', False), set( (
        ('negative', False),
       ) ),
     ),
    # Implications of extended_nonnegative = True:
    (('extended_nonnegative', True), set( (
        ('commutative', True),
        ('extended_negative', False),
        ('extended_real', True),
        ('imaginary', False),
        ('negative', False),
       ) ),
     ),
    # Implications of extended_nonnegative = False:
    (('extended_nonnegative', False), set( (
        ('composite', False),
        ('extended_positive', False),
        ('nonnegative', False),
        ('positive', False),
        ('prime', False),
        ('zero', False),
       ) ),
     ),
    # Implications of extended_nonpositive = True:
    (('extended_nonpositive', True), set( (
        ('commutative', True),
        ('composite', False),
        ('extended_positive', False),
        ('extended_real', True),
        ('imaginary', False),
        ('positive', False),
        ('prime', False),
       ) ),
     ),
    # Implications of extended_nonpositive = False:
    (('extended_nonpositive', False), set( (
        ('extended_negative', False),
        ('negative', False),
        ('nonpositive', False),
        ('zero', False),
       ) ),
     ),
    # Implications of extended_nonzero = True:
    (('extended_nonzero', True), set( (
        ('commutative', True),
        ('extended_real', True),
        ('imaginary', False),
        ('zero', False),
       ) ),
     ),
    # Implications of extended_nonzero = False:
    (('extended_nonzero', False), set( (
        ('composite', False),
        ('extended_negative', False),
        ('extended_positive', False),
        ('negative', False),
        ('nonzero', False),
        ('positive', False),
        ('prime', False),
       ) ),
     ),
    # Implications of extended_positive = True:
    (('extended_positive', True), set( (
        ('commutative', True),
        ('extended_negative', False),
        ('extended_nonnegative', True),
        ('extended_nonpositive', False),
        ('extended_nonzero', True),
        ('extended_real', True),
        ('imaginary', False),
        ('negative', False),
        ('nonpositive', False),
        ('zero', False),
       ) ),
     ),
    # Implications of extended_positive = False:
    (('extended_positive', False), set( (
        ('composite', False),
        ('positive', False),
        ('prime', False),
       ) ),
     ),
    # Implications of extended_real = True:
    (('extended_real', True), set( (
        ('commutative', True),
        ('imaginary', False),
       ) ),
     ),
    # Implications of extended_real = False:
    (('extended_real', False), set( (
        ('composite', False),
        ('even', False),
        ('extended_negative', False),
        ('extended_nonnegative', False),
        ('extended_nonpositive', False),
        ('extended_nonzero', False),
        ('extended_positive', False),
        ('integer', False),
        ('irrational', False),
        ('negative', False),
        ('noninteger', False),
        ('nonnegative', False),
        ('nonpositive', False),
        ('nonzero', False),
        ('odd', False),
        ('positive', False),
        ('prime', False),
        ('rational', False),
        ('real', False),
        ('zero', False),
       ) ),
     ),
    # Implications of finite = True:
    (('finite', True), set( (
        ('infinite', False),
       ) ),
     ),
    # Implications of finite = False:
    (('finite', False), set( (
        ('algebraic', False),
        ('complex', False),
        ('composite', False),
        ('even', False),
        ('imaginary', False),
        ('infinite', True),
        ('integer', False),
        ('irrational', False),
        ('negative', False),
        ('nonnegative', False),
        ('nonpositive', False),
        ('nonzero', False),
        ('odd', False),
        ('positive', False),
        ('prime', False),
        ('rational', False),
        ('real', False),
        ('transcendental', False),
        ('zero', False),
       ) ),
     ),
    # Implications of hermitian = True:
    (('hermitian', True), set( (
       ) ),
     ),
    # Implications of hermitian = False:
    (('hermitian', False), set( (
        ('composite', False),
        ('even', False),
        ('integer', False),
        ('irrational', False),
        ('negative', False),
        ('nonnegative', False),
        ('nonpositive', False),
        ('nonzero', False),
        ('odd', False),
        ('positive', False),
        ('prime', False),
        ('rational', False),
        ('real', False),
        ('zero', False),
       ) ),
     ),
    # Implications of imaginary = True:
    (('imaginary', True), set( (
        ('antihermitian', True),
        ('commutative', True),
        ('complex', True),
        ('composite', False),
        ('even', False),
        ('extended_negative', False),
        ('extended_nonnegative', False),
        ('extended_nonpositive', False),
        ('extended_nonzero', False),
        ('extended_positive', False),
        ('extended_real', False),
        ('finite', True),
        ('infinite', False),
        ('integer', False),
        ('irrational', False),
        ('negative', False),
        ('noninteger', False),
        ('nonnegative', False),
        ('nonpositive', False),
        ('nonzero', False),
        ('odd', False),
        ('positive', False),
        ('prime', False),
        ('rational', False),
        ('real', False),
        ('zero', False),
       ) ),
     ),
    # Implications of imaginary = False:
    (('imaginary', False), set( (
       ) ),
     ),
    # Implications of infinite = True:
    (('infinite', True), set( (
        ('algebraic', False),
        ('complex', False),
        ('composite', False),
        ('even', False),
        ('finite', False),
        ('imaginary', False),
        ('integer', False),
        ('irrational', False),
        ('negative', False),
        ('nonnegative', False),
        ('nonpositive', False),
        ('nonzero', False),
        ('odd', False),
        ('positive', False),
        ('prime', False),
        ('rational', False),
        ('real', False),
        ('transcendental', False),
        ('zero', False),
       ) ),
     ),
    # Implications of infinite = False:
    (('infinite', False), set( (
        ('finite', True),
       ) ),
     ),
    # Implications of integer = True:
    (('integer', True), set( (
        ('algebraic', True),
        ('commutative', True),
        ('complex', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('irrational', False),
        ('noninteger', False),
        ('rational', True),
        ('real', True),
        ('transcendental', False),
       ) ),
     ),
    # Implications of integer = False:
    (('integer', False), set( (
        ('composite', False),
        ('even', False),
        ('odd', False),
        ('prime', False),
        ('zero', False),
       ) ),
     ),
    # Implications of irrational = True:
    (('irrational', True), set( (
        ('commutative', True),
        ('complex', True),
        ('composite', False),
        ('even', False),
        ('extended_nonzero', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('integer', False),
        ('noninteger', True),
        ('nonzero', True),
        ('odd', False),
        ('prime', False),
        ('rational', False),
        ('real', True),
        ('zero', False),
       ) ),
     ),
    # Implications of irrational = False:
    (('irrational', False), set( (
       ) ),
     ),
    # Implications of negative = True:
    (('negative', True), set( (
        ('commutative', True),
        ('complex', True),
        ('composite', False),
        ('extended_negative', True),
        ('extended_nonnegative', False),
        ('extended_nonpositive', True),
        ('extended_nonzero', True),
        ('extended_positive', False),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('nonnegative', False),
        ('nonpositive', True),
        ('nonzero', True),
        ('positive', False),
        ('prime', False),
        ('real', True),
        ('zero', False),
       ) ),
     ),
    # Implications of negative = False:
    (('negative', False), set( (
       ) ),
     ),
    # Implications of noninteger = True:
    (('noninteger', True), set( (
        ('commutative', True),
        ('composite', False),
        ('even', False),
        ('extended_nonzero', True),
        ('extended_real', True),
        ('imaginary', False),
        ('integer', False),
        ('odd', False),
        ('prime', False),
        ('zero', False),
       ) ),
     ),
    # Implications of noninteger = False:
    (('noninteger', False), set( (
       ) ),
     ),
    # Implications of nonnegative = True:
    (('nonnegative', True), set( (
        ('commutative', True),
        ('complex', True),
        ('extended_negative', False),
        ('extended_nonnegative', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('negative', False),
        ('real', True),
       ) ),
     ),
    # Implications of nonnegative = False:
    (('nonnegative', False), set( (
        ('composite', False),
        ('positive', False),
        ('prime', False),
        ('zero', False),
       ) ),
     ),
    # Implications of nonpositive = True:
    (('nonpositive', True), set( (
        ('commutative', True),
        ('complex', True),
        ('composite', False),
        ('extended_nonpositive', True),
        ('extended_positive', False),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('positive', False),
        ('prime', False),
        ('real', True),
       ) ),
     ),
    # Implications of nonpositive = False:
    (('nonpositive', False), set( (
        ('negative', False),
        ('zero', False),
       ) ),
     ),
    # Implications of nonzero = True:
    (('nonzero', True), set( (
        ('commutative', True),
        ('complex', True),
        ('extended_nonzero', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('real', True),
        ('zero', False),
       ) ),
     ),
    # Implications of nonzero = False:
    (('nonzero', False), set( (
        ('composite', False),
        ('negative', False),
        ('positive', False),
        ('prime', False),
       ) ),
     ),
    # Implications of odd = True:
    (('odd', True), set( (
        ('algebraic', True),
        ('commutative', True),
        ('complex', True),
        ('even', False),
        ('extended_nonzero', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('integer', True),
        ('irrational', False),
        ('noninteger', False),
        ('nonzero', True),
        ('rational', True),
        ('real', True),
        ('transcendental', False),
        ('zero', False),
       ) ),
     ),
    # Implications of odd = False:
    (('odd', False), set( (
       ) ),
     ),
    # Implications of positive = True:
    (('positive', True), set( (
        ('commutative', True),
        ('complex', True),
        ('extended_negative', False),
        ('extended_nonnegative', True),
        ('extended_nonpositive', False),
        ('extended_nonzero', True),
        ('extended_positive', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('negative', False),
        ('nonnegative', True),
        ('nonpositive', False),
        ('nonzero', True),
        ('real', True),
        ('zero', False),
       ) ),
     ),
    # Implications of positive = False:
    (('positive', False), set( (
        ('composite', False),
        ('prime', False),
       ) ),
     ),
    # Implications of prime = True:
    (('prime', True), set( (
        ('algebraic', True),
        ('commutative', True),
        ('complex', True),
        ('composite', False),
        ('extended_negative', False),
        ('extended_nonnegative', True),
        ('extended_nonpositive', False),
        ('extended_nonzero', True),
        ('extended_positive', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('integer', True),
        ('irrational', False),
        ('negative', False),
        ('noninteger', False),
        ('nonnegative', True),
        ('nonpositive', False),
        ('nonzero', True),
        ('positive', True),
        ('rational', True),
        ('real', True),
        ('transcendental', False),
        ('zero', False),
       ) ),
     ),
    # Implications of prime = False:
    (('prime', False), set( (
       ) ),
     ),
    # Implications of rational = True:
    (('rational', True), set( (
        ('algebraic', True),
        ('commutative', True),
        ('complex', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('irrational', False),
        ('real', True),
        ('transcendental', False),
       ) ),
     ),
    # Implications of rational = False:
    (('rational', False), set( (
        ('composite', False),
        ('even', False),
        ('integer', False),
        ('odd', False),
        ('prime', False),
        ('zero', False),
       ) ),
     ),
    # Implications of real = True:
    (('real', True), set( (
        ('commutative', True),
        ('complex', True),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
       ) ),
     ),
    # Implications of real = False:
    (('real', False), set( (
        ('composite', False),
        ('even', False),
        ('integer', False),
        ('irrational', False),
        ('negative', False),
        ('nonnegative', False),
        ('nonpositive', False),
        ('nonzero', False),
        ('odd', False),
        ('positive', False),
        ('prime', False),
        ('rational', False),
        ('zero', False),
       ) ),
     ),
    # Implications of transcendental = True:
    (('transcendental', True), set( (
        ('algebraic', False),
        ('commutative', True),
        ('complex', True),
        ('composite', False),
        ('even', False),
        ('finite', True),
        ('infinite', False),
        ('integer', False),
        ('odd', False),
        ('prime', False),
        ('rational', False),
        ('zero', False),
       ) ),
     ),
    # Implications of transcendental = False:
    (('transcendental', False), set( (
       ) ),
     ),
    # Implications of zero = True:
    (('zero', True), set( (
        ('algebraic', True),
        ('commutative', True),
        ('complex', True),
        ('composite', False),
        ('even', True),
        ('extended_negative', False),
        ('extended_nonnegative', True),
        ('extended_nonpositive', True),
        ('extended_nonzero', False),
        ('extended_positive', False),
        ('extended_real', True),
        ('finite', True),
        ('hermitian', True),
        ('imaginary', False),
        ('infinite', False),
        ('integer', True),
        ('irrational', False),
        ('negative', False),
        ('noninteger', False),
        ('nonnegative', True),
        ('nonpositive', True),
        ('nonzero', False),
        ('odd', False),
        ('positive', False),
        ('prime', False),
        ('rational', True),
        ('real', True),
        ('transcendental', False),
       ) ),
     ),
    # Implications of zero = False:
    (('zero', False), set( (
       ) ),
     ),
 ] ) # full_implications


prereq = {

    # facts that could determine the value of algebraic
    'algebraic': {
        'commutative',
        'complex',
        'composite',
        'even',
        'finite',
        'infinite',
        'integer',
        'odd',
        'prime',
        'rational',
        'transcendental',
        'zero',
    },

    # facts that could determine the value of antihermitian
    'antihermitian': {
        'imaginary',
    },

    # facts that could determine the value of commutative
    'commutative': {
        'algebraic',
        'complex',
        'composite',
        'even',
        'extended_negative',
        'extended_nonnegative',
        'extended_nonpositive',
        'extended_nonzero',
        'extended_positive',
        'extended_real',
        'imaginary',
        'integer',
        'irrational',
        'negative',
        'noninteger',
        'nonnegative',
        'nonpositive',
        'nonzero',
        'odd',
        'positive',
        'prime',
        'rational',
        'real',
        'transcendental',
        'zero',
    },

    # facts that could determine the value of complex
    'complex': {
        'algebraic',
        'commutative',
        'composite',
        'even',
        'finite',
        'imaginary',
        'infinite',
        'integer',
        'irrational',
        'negative',
        'nonnegative',
        'nonpositive',
        'nonzero',
        'odd',
        'positive',
        'prime',
        'rational',
        'real',
        'transcendental',
        'zero',
    },

    # facts that could determine the value of composite
    'composite': {
        'algebraic',
        'commutative',
        'complex',
        'extended_negative',
        'extended_nonnegative',
        'extended_nonpositive',
        'extended_nonzero',
        'extended_positive',
        'extended_real',
        'finite',
        'hermitian',
        'imaginary',
        'infinite',
        'integer',
        'irrational',
        'negative',
        'noninteger',
        'nonnegative',
        'nonpositive',
        'nonzero',
        'positive',
        'prime',
        'rational',
        'real',
        'transcendental',
        'zero',
    },

    # facts that could determine the value of even
    'even': {
        'algebraic',
        'commutative',
        'complex',
        'extended_real',
        'finite',
        'hermitian',
        'imaginary',
        'infinite',
        'integer',
        'irrational',
        'noninteger',
        'odd',
        'rational',
        'real',
        'transcendental',
        'zero',
    },

    # facts that could determine the value of extended_negative
    'extended_negative': {
        'commutative',
        'composite',
        'extended_nonnegative',
        'extended_nonpositive',
        'extended_nonzero',
        'extended_positive',
        'extended_real',
        'imaginary',
        'negative',
        'nonnegative',
        'positive',
        'prime',
        'zero',
    },

    # facts that could determine the value of extended_nonnegative
    'extended_nonnegative': {
        'commutative',
        'composite',
        'extended_negative',
        'extended_positive',
        'extended_real',
        'imaginary',
        'negative',
        'nonnegative',
        'positive',
        'prime',
        'zero',
    },

    # facts that could determine the value of extended_nonpositive
    'extended_nonpositive': {
        'commutative',
        'composite',
        'extended_negative',
        'extended_positive',
        'extended_real',
        'imaginary',
        'negative',
        'nonpositive',
        'positive',
        'prime',
        'zero',
    },

    # facts that could determine the value of extended_nonzero
    'extended_nonzero': {
        'commutative',
        'composite',
        'extended_negative',
        'extended_positive',
        'extended_real',
        'imaginary',
        'irrational',
        'negative',
        'noninteger',
        'nonzero',
        'odd',
        'positive',
        'prime',
        'zero',
    },

    # facts that could determine the value of extended_positive
    'extended_positive': {
        'commutative',
        'composite',
        'extended_negative',
        'extended_nonnegative',
        'extended_nonpositive',
        'extended_nonzero',
        'extended_real',
        'imaginary',
        'negative',
        'nonpositive',
        'positive',
        'prime',
        'zero',
    },

    # facts that could determine the value of extended_real
    'extended_real': {
        'commutative',
        'composite',
        'even',
        'extended_negative',
        'extended_nonnegative',
        'extended_nonpositive',
        'extended_nonzero',
        'extended_positive',
        'imaginary',
        'integer',
        'irrational',
        'negative',
        'noninteger',
        'nonnegative',
        'nonpositive',
        'nonzero',
        'odd',
        'positive',
        'prime',
        'rational',
        'real',
        'zero',
    },

    # facts that could determine the value of finite
    'finite': {
        'algebraic',
        'complex',
        'composite',
        'even',
        'imaginary',
        'infinite',
        'integer',
        'irrational',
        'negative',
        'nonnegative',
        'nonpositive',
        'nonzero',
        'odd',
        'positive',
        'prime',
        'rational',
        'real',
        'transcendental',
        'zero',
    },

    # facts that could determine the value of hermitian
    'hermitian': {
        'composite',
        'even',
        'integer',
        'irrational',
        'negative',
        'nonnegative',
        'nonpositive',
        'nonzero',
        'odd',
        'positive',
        'prime',
        'rational',
        'real',
        'zero',
    },

    # facts that could determine the value of imaginary
    'imaginary': {
        'antihermitian',
        'commutative',
        'complex',
        'composite',
        'even',
        'extended_negative',
        'extended_nonnegative',
        'extended_nonpositive',
        'extended_nonzero',
        'extended_positive',
        'extended_real',
        'finite',
        'infinite',
        'integer',
        'irrational',
        'negative',
        'noninteger',
        'nonnegative',
        'nonpositive',
        'nonzero',
        'odd',
        'positive',
        'prime',
        'rational',
        'real',
        'zero',
    },

    # facts that could determine the value of infinite
    'infinite': {
        'algebraic',
        'complex',
        'composite',
        'even',
        'finite',
        'imaginary',
        'integer',
        'irrational',
        'negative',
        'nonnegative',
        'nonpositive',
        'nonzero',
        'odd',
        'positive',
        'prime',
        'rational',
        'real',
        'transcendental',
        'zero',
    },

    # facts that could determine the value of integer
    'integer': {
        'algebraic',
        'commutative',
        'complex',
        'composite',
        'even',
        'extended_real',
        'finite',
        'hermitian',
        'imaginary',
        'infinite',
        'irrational',
        'noninteger',
        'odd',
        'prime',
        'rational',
        'real',
        'transcendental',
        'zero',
    },

    # facts that could determine the value of irrational
    'irrational': {
        'commutative',
        'complex',
        'composite',
        'even',
        'extended_real',
        'finite',
        'hermitian',
        'imaginary',
        'infinite',
        'integer',
        'odd',
        'prime',
        'rational',
        'real',
        'zero',
    },

    # facts that could determine the value of negative
    'negative': {
        'commutative',
        'complex',
        'composite',
        'extended_negative',
        'extended_nonnegative',
        'extended_nonpositive',
        'extended_nonzero',
        'extended_positive',
        'extended_real',
        'finite',
        'hermitian',
        'imaginary',
        'infinite',
        'nonnegative',
        'nonpositive',
        'nonzero',
        'positive',
        'prime',
        'real',
        'zero',
    },

    # facts that could determine the value of noninteger
    'noninteger': {
        'commutative',
        'composite',
        'even',
        'extended_real',
        'imaginary',
        'integer',
        'irrational',
        'odd',
        'prime',
        'zero',
    },

    # facts that could determine the value of nonnegative
    'nonnegative': {
        'commutative',
        'complex',
        'composite',
        'extended_negative',
        'extended_nonnegative',
        'extended_real',
        'finite',
        'hermitian',
        'imaginary',
        'infinite',
        'negative',
        'positive',
        'prime',
        'real',
        'zero',
    },

    # facts that could determine the value of nonpositive
    'nonpositive': {
        'commutative',
        'complex',
        'composite',
        'extended_nonpositive',
        'extended_positive',
        'extended_real',
        'finite',
        'hermitian',
        'imaginary',
        'infinite',
        'negative',
        'positive',
        'prime',
        'real',
        'zero',
    },

    # facts that could determine the value of nonzero
    'nonzero': {
        'commutative',
        'complex',
        'composite',
        'extended_nonzero',
        'extended_real',
        'finite',
        'hermitian',
        'imaginary',
        'infinite',
        'irrational',
        'negative',
        'odd',
        'positive',
        'prime',
        'real',
        'zero',
    },

    # facts that could determine the value of odd
    'odd': {
        'algebraic',
        'commutative',
        'complex',
        'even',
        'extended_real',
        'finite',
        'hermitian',
        'imaginary',
        'infinite',
        'integer',
        'irrational',
        'noninteger',
        'rational',
        'real',
        'transcendental',
        'zero',
    },

    # facts that could determine the value of positive
    'positive': {
        'commutative',
        'complex',
        'composite',
        'extended_negative',
        'extended_nonnegative',
        'extended_nonpositive',
        'extended_nonzero',
        'extended_positive',
        'extended_real',
        'finite',
        'hermitian',
        'imaginary',
        'infinite',
        'negative',
        'nonnegative',
        'nonpositive',
        'nonzero',
        'prime',
        'real',
        'zero',
    },

    # facts that could determine the value of prime
    'prime': {
        'algebraic',
        'commutative',
        'complex',
        'composite',
        'extended_negative',
        'extended_nonnegative',
        'extended_nonpositive',
        'extended_nonzero',
        'extended_positive',
        'extended_real',
        'finite',
        'hermitian',
        'imaginary',
        'infinite',
        'integer',
        'irrational',
        'negative',
        'noninteger',
        'nonnegative',
        'nonpositive',
        'nonzero',
        'positive',
        'rational',
        'real',
        'transcendental',
        'zero',
    },

    # facts that could determine the value of rational
    'rational': {
        'algebraic',
        'commutative',
        'complex',
        'composite',
        'even',
        'extended_real',
        'finite',
        'hermitian',
        'imaginary',
        'infinite',
        'integer',
        'irrational',
        'odd',
        'prime',
        'real',
        'transcendental',
        'zero',
    },

    # facts that could determine the value of real
    'real': {
        'commutative',
        'complex',
        'composite',
        'even',
        'extended_real',
        'finite',
        'hermitian',
        'imaginary',
        'infinite',
        'integer',
        'irrational',
        'negative',
        'nonnegative',
        'nonpositive',
        'nonzero',
        'odd',
        'positive',
        'prime',
        'rational',
        'zero',
    },

    # facts that could determine the value of transcendental
    'transcendental': {
        'algebraic',
        'commutative',
        'complex',
        'composite',
        'even',
        'finite',
        'infinite',
        'integer',
        'odd',
        'prime',
        'rational',
        'zero',
    },

    # facts that could determine the value of zero
    'zero': {
        'algebraic',
        'commutative',
        'complex',
        'composite',
        'even',
        'extended_negative',
        'extended_nonnegative',
        'extended_nonpositive',
        'extended_nonzero',
        'extended_positive',
        'extended_real',
        'finite',
        'hermitian',
        'imaginary',
        'infinite',
        'integer',
        'irrational',
        'negative',
        'noninteger',
        'nonnegative',
        'nonpositive',
        'nonzero',
        'odd',
        'positive',
        'prime',
        'rational',
        'real',
        'transcendental',
    },

} # prereq


# Note: the order of the beta rules is used in the beta_triggers
beta_rules = [

    # Rules implying composite = True
    ({('even', True), ('positive', True), ('prime', False)},
        ('composite', True)),

    # Rules implying even = False
    ({('composite', False), ('positive', True), ('prime', False)},
        ('even', False)),

    # Rules implying even = True
    ({('integer', True), ('odd', False)},
        ('even', True)),

    # Rules implying extended_negative = True
    ({('extended_positive', False), ('extended_real', True), ('zero', False)},
        ('extended_negative', True)),
    ({('extended_nonpositive', True), ('extended_nonzero', True)},
        ('extended_negative', True)),

    # Rules implying extended_nonnegative = True
    ({('extended_negative', False), ('extended_real', True)},
        ('extended_nonnegative', True)),

    # Rules implying extended_nonpositive = True
    ({('extended_positive', False), ('extended_real', True)},
        ('extended_nonpositive', True)),

    # Rules implying extended_nonzero = True
    ({('extended_real', True), ('zero', False)},
        ('extended_nonzero', True)),

    # Rules implying extended_positive = True
    ({('extended_negative', False), ('extended_real', True), ('zero', False)},
        ('extended_positive', True)),
    ({('extended_nonnegative', True), ('extended_nonzero', True)},
        ('extended_positive', True)),

    # Rules implying extended_real = False
    ({('infinite', False), ('real', False)},
        ('extended_real', False)),
    ({('extended_negative', False), ('extended_positive', False), ('zero', False)},
        ('extended_real', False)),

    # Rules implying infinite = True
    ({('extended_real', True), ('real', False)},
        ('infinite', True)),

    # Rules implying irrational = True
    ({('rational', False), ('real', True)},
        ('irrational', True)),

    # Rules implying negative = True
    ({('positive', False), ('real', True), ('zero', False)},
        ('negative', True)),
    ({('nonpositive', True), ('nonzero', True)},
        ('negative', True)),
    ({('extended_negative', True), ('finite', True)},
        ('negative', True)),

    # Rules implying noninteger = True
    ({('extended_real', True), ('integer', False)},
        ('noninteger', True)),

    # Rules implying nonnegative = True
    ({('negative', False), ('real', True)},
        ('nonnegative', True)),
    ({('extended_nonnegative', True), ('finite', True)},
        ('nonnegative', True)),

    # Rules implying nonpositive = True
    ({('positive', False), ('real', True)},
        ('nonpositive', True)),
    ({('extended_nonpositive', True), ('finite', True)},
        ('nonpositive', True)),

    # Rules implying nonzero = True
    ({('extended_nonzero', True), ('finite', True)},
        ('nonzero', True)),

    # Rules implying odd = True
    ({('even', False), ('integer', True)},
        ('odd', True)),

    # Rules implying positive = False
    ({('composite', False), ('even', True), ('prime', False)},
        ('positive', False)),

    # Rules implying positive = True
    ({('negative', False), ('real', True), ('zero', False)},
        ('positive', True)),
    ({('nonnegative', True), ('nonzero', True)},
        ('positive', True)),
    ({('extended_positive', True), ('finite', True)},
        ('positive', True)),

    # Rules implying prime = True
    ({('composite', False), ('even', True), ('positive', True)},
        ('prime', True)),

    # Rules implying real = False
    ({('negative', False), ('positive', False), ('zero', False)},
        ('real', False)),

    # Rules implying real = True
    ({('extended_real', True), ('infinite', False)},
        ('real', True)),
    ({('extended_real', True), ('finite', True)},
        ('real', True)),

    # Rules implying transcendental = True
    ({('algebraic', False), ('complex', True)},
        ('transcendental', True)),

    # Rules implying zero = True
    ({('extended_negative', False), ('extended_positive', False), ('extended_real', True)},
        ('zero', True)),
    ({('negative', False), ('positive', False), ('real', True)},
        ('zero', True)),
    ({('extended_nonnegative', True), ('extended_nonpositive', True)},
        ('zero', True)),
    ({('nonnegative', True), ('nonpositive', True)},
        ('zero', True)),

] # beta_rules
beta_triggers = {
    ('algebraic', False): [32, 11, 3, 8, 29, 14, 25, 13, 17, 7],
    ('algebraic', True): [10, 30, 31, 27, 16, 21, 19, 22],
    ('antihermitian', False): [],
    ('commutative', False): [],
    ('complex', False): [10, 12, 11, 3, 8, 17, 7],
    ('complex', True): [32, 10, 30, 31, 27, 16, 21, 19, 22],
    ('composite', False): [1, 28, 24],
    ('composite', True): [23, 2],
    ('even', False): [23, 11, 3, 8, 29, 14, 25, 7],
    ('even', True): [3, 33, 8, 6, 5, 14, 34, 25, 20, 18, 27, 16, 21, 19, 22, 0, 28, 24, 7],
    ('extended_negative', False): [11, 33, 8, 5, 29, 34, 25, 18],
    ('extended_negative', True): [30, 12, 31, 29, 14, 20, 16, 21, 22, 17],
    ('extended_nonnegative', False): [11, 3, 6, 29, 14, 20, 7],
    ('extended_nonnegative', True): [30, 12, 31, 33, 8, 9, 6, 29, 34, 25, 18, 19, 35, 17, 7],
    ('extended_nonpositive', False): [11, 8, 5, 29, 25, 18, 7],
    ('extended_nonpositive', True): [30, 12, 31, 3, 33, 4, 5, 29, 14, 34, 20, 21, 35, 17, 7],
    ('extended_nonzero', False): [11, 33, 6, 5, 29, 34, 20, 18],
    ('extended_nonzero', True): [30, 12, 31, 3, 8, 4, 9, 6, 5, 29, 14, 25, 22, 17],
    ('extended_positive', False): [11, 3, 33, 6, 29, 14, 34, 20],
    ('extended_positive', True): [30, 12, 31, 29, 25, 18, 27, 19, 22, 17],
    ('extended_real', False): [],
    ('extended_real', True): [30, 12, 31, 3, 33, 8, 6, 5, 17, 7],
    ('finite', False): [11, 3, 8, 17, 7],
    ('finite', True): [10, 30, 31, 27, 16, 21, 19, 22],
    ('hermitian', False): [10, 12, 11, 3, 8, 17, 7],
    ('imaginary', True): [32],
    ('infinite', False): [10, 30, 31, 27, 16, 21, 19, 22],
    ('infinite', True): [11, 3, 8, 17, 7],
    ('integer', False): [11, 3, 8, 29, 14, 25, 17, 7],
    ('integer', True): [23, 2, 3, 33, 8, 6, 5, 14, 34, 25, 20, 18, 27, 16, 21, 19, 22, 7],
    ('irrational', True): [32, 3, 8, 4, 9, 6, 5, 14, 25, 15, 26, 20, 18, 27, 16, 21, 19],
    ('negative', False): [29, 34, 25, 18],
    ('negative', True): [32, 13, 17],
    ('noninteger', True): [30, 12, 31, 3, 8, 4, 9, 6, 5, 29, 14, 25, 22],
    ('nonnegative', False): [11, 3, 8, 29, 14, 20, 7],
    ('nonnegative', True): [32, 33, 8, 9, 6, 34, 25, 26, 20, 27, 21, 22, 35, 36, 13, 17, 7],
    ('nonpositive', False): [11, 3, 8, 29, 25, 18, 7],
    ('nonpositive', True): [32, 3, 33, 4, 5, 14, 34, 15, 18, 16, 19, 22, 35, 36, 13, 17, 7],
    ('nonzero', False): [29, 34, 20, 18],
    ('nonzero', True): [32, 3, 8, 4, 9, 6, 5, 14, 25, 15, 26, 20, 18, 27, 16, 21, 19, 13, 17],
    ('odd', False): [2],
    ('odd', True): [3, 8, 4, 9, 6, 5, 14, 25, 15, 26, 20, 18, 27, 16, 21, 19],
    ('positive', False): [29, 14, 34, 20],
    ('positive', True): [32, 0, 1, 28, 13, 17],
    ('prime', False): [0, 1, 24],
    ('prime', True): [23, 2],
    ('rational', False): [11, 3, 8, 29, 14, 25, 13, 17, 7],
    ('rational', True): [3, 33, 8, 6, 5, 14, 34, 25, 20, 18, 27, 16, 21, 19, 22, 17, 7],
    ('real', False): [10, 12, 11, 3, 8, 17, 7],
    ('real', True): [32, 3, 33, 8, 6, 5, 14, 34, 25, 20, 18, 27, 16, 21, 19, 22, 13, 17, 7],
    ('transcendental', True): [10, 30, 31, 11, 3, 8, 29, 14, 25, 27, 16, 21, 19, 22, 13, 17, 7],
    ('zero', False): [11, 3, 8, 29, 14, 25, 7],
    ('zero', True): [],
} # beta_triggers


generated_assumptions = {'defined_facts': defined_facts, 'full_implications': full_implications,
               'prereq': prereq, 'beta_rules': beta_rules, 'beta_triggers': beta_triggers}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dateutil\parser\_parser.py ===
# -*- coding: utf-8 -*-
"""
This module offers a generic date/time string parser which is able to parse
most known formats to represent a date and/or time.

This module attempts to be forgiving with regards to unlikely input formats,
returning a datetime object even for dates which are ambiguous. If an element
of a date/time stamp is omitted, the following rules are applied:

- If AM or PM is left unspecified, a 24-hour clock is assumed, however, an hour
  on a 12-hour clock (``0 <= hour <= 12``) *must* be specified if AM or PM is
  specified.
- If a time zone is omitted, a timezone-naive datetime is returned.

If any other elements are missing, they are taken from the
:class:`datetime.datetime` object passed to the parameter ``default``. If this
results in a day number exceeding the valid number of days per month, the
value falls back to the end of the month.

Additional resources about date/time string formats can be found below:

- `A summary of the international standard date and time notation
  <https://www.cl.cam.ac.uk/~mgk25/iso-time.html>`_
- `W3C Date and Time Formats <https://www.w3.org/TR/NOTE-datetime>`_
- `Time Formats (Planetary Rings Node) <https://pds-rings.seti.org:443/tools/time_formats.html>`_
- `CPAN ParseDate module
  <https://metacpan.org/pod/release/MUIR/Time-modules-2013.0912/lib/Time/ParseDate.pm>`_
- `Java SimpleDateFormat Class
  <https://docs.oracle.com/javase/6/docs/api/java/text/SimpleDateFormat.html>`_
"""
from __future__ import unicode_literals

import datetime
import re
import string
import time
import warnings

from calendar import monthrange
from io import StringIO

import six
from six import integer_types, text_type

from decimal import Decimal

from warnings import warn

from .. import relativedelta
from .. import tz

__all__ = ["parse", "parserinfo", "ParserError"]


# TODO: pandas.core.tools.datetimes imports this explicitly.  Might be worth
# making public and/or figuring out if there is something we can
# take off their plate.
class _timelex(object):
    # Fractional seconds are sometimes split by a comma
    _split_decimal = re.compile("([.,])")

    def __init__(self, instream):
        if isinstance(instream, (bytes, bytearray)):
            instream = instream.decode()

        if isinstance(instream, text_type):
            instream = StringIO(instream)
        elif getattr(instream, 'read', None) is None:
            raise TypeError('Parser must be a string or character stream, not '
                            '{itype}'.format(itype=instream.__class__.__name__))

        self.instream = instream
        self.charstack = []
        self.tokenstack = []
        self.eof = False

    def get_token(self):
        """
        This function breaks the time string into lexical units (tokens), which
        can be parsed by the parser. Lexical units are demarcated by changes in
        the character set, so any continuous string of letters is considered
        one unit, any continuous string of numbers is considered one unit.

        The main complication arises from the fact that dots ('.') can be used
        both as separators (e.g. "Sep.20.2009") or decimal points (e.g.
        "4:30:21.447"). As such, it is necessary to read the full context of
        any dot-separated strings before breaking it into tokens; as such, this
        function maintains a "token stack", for when the ambiguous context
        demands that multiple tokens be parsed at once.
        """
        if self.tokenstack:
            return self.tokenstack.pop(0)

        seenletters = False
        token = None
        state = None

        while not self.eof:
            # We only realize that we've reached the end of a token when we
            # find a character that's not part of the current token - since
            # that character may be part of the next token, it's stored in the
            # charstack.
            if self.charstack:
                nextchar = self.charstack.pop(0)
            else:
                nextchar = self.instream.read(1)
                while nextchar == '\x00':
                    nextchar = self.instream.read(1)

            if not nextchar:
                self.eof = True
                break
            elif not state:
                # First character of the token - determines if we're starting
                # to parse a word, a number or something else.
                token = nextchar
                if self.isword(nextchar):
                    state = 'a'
                elif self.isnum(nextchar):
                    state = '0'
                elif self.isspace(nextchar):
                    token = ' '
                    break  # emit token
                else:
                    break  # emit token
            elif state == 'a':
                # If we've already started reading a word, we keep reading
                # letters until we find something that's not part of a word.
                seenletters = True
                if self.isword(nextchar):
                    token += nextchar
                elif nextchar == '.':
                    token += nextchar
                    state = 'a.'
                else:
                    self.charstack.append(nextchar)
                    break  # emit token
            elif state == '0':
                # If we've already started reading a number, we keep reading
                # numbers until we find something that doesn't fit.
                if self.isnum(nextchar):
                    token += nextchar
                elif nextchar == '.' or (nextchar == ',' and len(token) >= 2):
                    token += nextchar
                    state = '0.'
                else:
                    self.charstack.append(nextchar)
                    break  # emit token
            elif state == 'a.':
                # If we've seen some letters and a dot separator, continue
                # parsing, and the tokens will be broken up later.
                seenletters = True
                if nextchar == '.' or self.isword(nextchar):
                    token += nextchar
                elif self.isnum(nextchar) and token[-1] == '.':
                    token += nextchar
                    state = '0.'
                else:
                    self.charstack.append(nextchar)
                    break  # emit token
            elif state == '0.':
                # If we've seen at least one dot separator, keep going, we'll
                # break up the tokens later.
                if nextchar == '.' or self.isnum(nextchar):
                    token += nextchar
                elif self.isword(nextchar) and token[-1] == '.':
                    token += nextchar
                    state = 'a.'
                else:
                    self.charstack.append(nextchar)
                    break  # emit token

        if (state in ('a.', '0.') and (seenletters or token.count('.') > 1 or
                                       token[-1] in '.,')):
            l = self._split_decimal.split(token)
            token = l[0]
            for tok in l[1:]:
                if tok:
                    self.tokenstack.append(tok)

        if state == '0.' and token.count('.') == 0:
            token = token.replace(',', '.')

        return token

    def __iter__(self):
        return self

    def __next__(self):
        token = self.get_token()
        if token is None:
            raise StopIteration

        return token

    def next(self):
        return self.__next__()  # Python 2.x support

    @classmethod
    def split(cls, s):
        return list(cls(s))

    @classmethod
    def isword(cls, nextchar):
        """ Whether or not the next character is part of a word """
        return nextchar.isalpha()

    @classmethod
    def isnum(cls, nextchar):
        """ Whether the next character is part of a number """
        return nextchar.isdigit()

    @classmethod
    def isspace(cls, nextchar):
        """ Whether the next character is whitespace """
        return nextchar.isspace()


class _resultbase(object):

    def __init__(self):
        for attr in self.__slots__:
            setattr(self, attr, None)

    def _repr(self, classname):
        l = []
        for attr in self.__slots__:
            value = getattr(self, attr)
            if value is not None:
                l.append("%s=%s" % (attr, repr(value)))
        return "%s(%s)" % (classname, ", ".join(l))

    def __len__(self):
        return (sum(getattr(self, attr) is not None
                    for attr in self.__slots__))

    def __repr__(self):
        return self._repr(self.__class__.__name__)


class parserinfo(object):
    """
    Class which handles what inputs are accepted. Subclass this to customize
    the language and acceptable values for each parameter.

    :param dayfirst:
        Whether to interpret the first value in an ambiguous 3-integer date
        (e.g. 01/05/09) as the day (``True``) or month (``False``). If
        ``yearfirst`` is set to ``True``, this distinguishes between YDM
        and YMD. Default is ``False``.

    :param yearfirst:
        Whether to interpret the first value in an ambiguous 3-integer date
        (e.g. 01/05/09) as the year. If ``True``, the first number is taken
        to be the year, otherwise the last number is taken to be the year.
        Default is ``False``.
    """

    # m from a.m/p.m, t from ISO T separator
    JUMP = [" ", ".", ",", ";", "-", "/", "'",
            "at", "on", "and", "ad", "m", "t", "of",
            "st", "nd", "rd", "th"]

    WEEKDAYS = [("Mon", "Monday"),
                ("Tue", "Tuesday"),     # TODO: "Tues"
                ("Wed", "Wednesday"),
                ("Thu", "Thursday"),    # TODO: "Thurs"
                ("Fri", "Friday"),
                ("Sat", "Saturday"),
                ("Sun", "Sunday")]
    MONTHS = [("Jan", "January"),
              ("Feb", "February"),      # TODO: "Febr"
              ("Mar", "March"),
              ("Apr", "April"),
              ("May", "May"),
              ("Jun", "June"),
              ("Jul", "July"),
              ("Aug", "August"),
              ("Sep", "Sept", "September"),
              ("Oct", "October"),
              ("Nov", "November"),
              ("Dec", "December")]
    HMS = [("h", "hour", "hours"),
           ("m", "minute", "minutes"),
           ("s", "second", "seconds")]
    AMPM = [("am", "a"),
            ("pm", "p")]
    UTCZONE = ["UTC", "GMT", "Z", "z"]
    PERTAIN = ["of"]
    TZOFFSET = {}
    # TODO: ERA = ["AD", "BC", "CE", "BCE", "Stardate",
    #              "Anno Domini", "Year of Our Lord"]

    def __init__(self, dayfirst=False, yearfirst=False):
        self._jump = self._convert(self.JUMP)
        self._weekdays = self._convert(self.WEEKDAYS)
        self._months = self._convert(self.MONTHS)
        self._hms = self._convert(self.HMS)
        self._ampm = self._convert(self.AMPM)
        self._utczone = self._convert(self.UTCZONE)
        self._pertain = self._convert(self.PERTAIN)

        self.dayfirst = dayfirst
        self.yearfirst = yearfirst

        self._year = time.localtime().tm_year
        self._century = self._year // 100 * 100

    def _convert(self, lst):
        dct = {}
        for i, v in enumerate(lst):
            if isinstance(v, tuple):
                for v in v:
                    dct[v.lower()] = i
            else:
                dct[v.lower()] = i
        return dct

    def jump(self, name):
        return name.lower() in self._jump

    def weekday(self, name):
        try:
            return self._weekdays[name.lower()]
        except KeyError:
            pass
        return None

    def month(self, name):
        try:
            return self._months[name.lower()] + 1
        except KeyError:
            pass
        return None

    def hms(self, name):
        try:
            return self._hms[name.lower()]
        except KeyError:
            return None

    def ampm(self, name):
        try:
            return self._ampm[name.lower()]
        except KeyError:
            return None

    def pertain(self, name):
        return name.lower() in self._pertain

    def utczone(self, name):
        return name.lower() in self._utczone

    def tzoffset(self, name):
        if name in self._utczone:
            return 0

        return self.TZOFFSET.get(name)

    def convertyear(self, year, century_specified=False):
        """
        Converts two-digit years to year within [-50, 49]
        range of self._year (current local time)
        """

        # Function contract is that the year is always positive
        assert year >= 0

        if year < 100 and not century_specified:
            # assume current century to start
            year += self._century

            if year >= self._year + 50:  # if too far in future
                year -= 100
            elif year < self._year - 50:  # if too far in past
                year += 100

        return year

    def validate(self, res):
        # move to info
        if res.year is not None:
            res.year = self.convertyear(res.year, res.century_specified)

        if ((res.tzoffset == 0 and not res.tzname) or
             (res.tzname == 'Z' or res.tzname == 'z')):
            res.tzname = "UTC"
            res.tzoffset = 0
        elif res.tzoffset != 0 and res.tzname and self.utczone(res.tzname):
            res.tzoffset = 0
        return True


class _ymd(list):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.century_specified = False
        self.dstridx = None
        self.mstridx = None
        self.ystridx = None

    @property
    def has_year(self):
        return self.ystridx is not None

    @property
    def has_month(self):
        return self.mstridx is not None

    @property
    def has_day(self):
        return self.dstridx is not None

    def could_be_day(self, value):
        if self.has_day:
            return False
        elif not self.has_month:
            return 1 <= value <= 31
        elif not self.has_year:
            # Be permissive, assume leap year
            month = self[self.mstridx]
            return 1 <= value <= monthrange(2000, month)[1]
        else:
            month = self[self.mstridx]
            year = self[self.ystridx]
            return 1 <= value <= monthrange(year, month)[1]

    def append(self, val, label=None):
        if hasattr(val, '__len__'):
            if val.isdigit() and len(val) > 2:
                self.century_specified = True
                if label not in [None, 'Y']:  # pragma: no cover
                    raise ValueError(label)
                label = 'Y'
        elif val > 100:
            self.century_specified = True
            if label not in [None, 'Y']:  # pragma: no cover
                raise ValueError(label)
            label = 'Y'

        super(self.__class__, self).append(int(val))

        if label == 'M':
            if self.has_month:
                raise ValueError('Month is already set')
            self.mstridx = len(self) - 1
        elif label == 'D':
            if self.has_day:
                raise ValueError('Day is already set')
            self.dstridx = len(self) - 1
        elif label == 'Y':
            if self.has_year:
                raise ValueError('Year is already set')
            self.ystridx = len(self) - 1

    def _resolve_from_stridxs(self, strids):
        """
        Try to resolve the identities of year/month/day elements using
        ystridx, mstridx, and dstridx, if enough of these are specified.
        """
        if len(self) == 3 and len(strids) == 2:
            # we can back out the remaining stridx value
            missing = [x for x in range(3) if x not in strids.values()]
            key = [x for x in ['y', 'm', 'd'] if x not in strids]
            assert len(missing) == len(key) == 1
            key = key[0]
            val = missing[0]
            strids[key] = val

        assert len(self) == len(strids)  # otherwise this should not be called
        out = {key: self[strids[key]] for key in strids}
        return (out.get('y'), out.get('m'), out.get('d'))

    def resolve_ymd(self, yearfirst, dayfirst):
        len_ymd = len(self)
        year, month, day = (None, None, None)

        strids = (('y', self.ystridx),
                  ('m', self.mstridx),
                  ('d', self.dstridx))

        strids = {key: val for key, val in strids if val is not None}
        if (len(self) == len(strids) > 0 or
                (len(self) == 3 and len(strids) == 2)):
            return self._resolve_from_stridxs(strids)

        mstridx = self.mstridx

        if len_ymd > 3:
            raise ValueError("More than three YMD values")
        elif len_ymd == 1 or (mstridx is not None and len_ymd == 2):
            # One member, or two members with a month string
            if mstridx is not None:
                month = self[mstridx]
                # since mstridx is 0 or 1, self[mstridx-1] always
                # looks up the other element
                other = self[mstridx - 1]
            else:
                other = self[0]

            if len_ymd > 1 or mstridx is None:
                if other > 31:
                    year = other
                else:
                    day = other

        elif len_ymd == 2:
            # Two members with numbers
            if self[0] > 31:
                # 99-01
                year, month = self
            elif self[1] > 31:
                # 01-99
                month, year = self
            elif dayfirst and self[1] <= 12:
                # 13-01
                day, month = self
            else:
                # 01-13
                month, day = self

        elif len_ymd == 3:
            # Three members
            if mstridx == 0:
                if self[1] > 31:
                    # Apr-2003-25
                    month, year, day = self
                else:
                    month, day, year = self
            elif mstridx == 1:
                if self[0] > 31 or (yearfirst and self[2] <= 31):
                    # 99-Jan-01
                    year, month, day = self
                else:
                    # 01-Jan-01
                    # Give precedence to day-first, since
                    # two-digit years is usually hand-written.
                    day, month, year = self

            elif mstridx == 2:
                # WTF!?
                if self[1] > 31:
                    # 01-99-Jan
                    day, year, month = self
                else:
                    # 99-01-Jan
                    year, day, month = self

            else:
                if (self[0] > 31 or
                    self.ystridx == 0 or
                        (yearfirst and self[1] <= 12 and self[2] <= 31)):
                    # 99-01-01
                    if dayfirst and self[2] <= 12:
                        year, day, month = self
                    else:
                        year, month, day = self
                elif self[0] > 12 or (dayfirst and self[1] <= 12):
                    # 13-01-01
                    day, month, year = self
                else:
                    # 01-13-01
                    month, day, year = self

        return year, month, day


class parser(object):
    def __init__(self, info=None):
        self.info = info or parserinfo()

    def parse(self, timestr, default=None,
              ignoretz=False, tzinfos=None, **kwargs):
        """
        Parse the date/time string into a :class:`datetime.datetime` object.

        :param timestr:
            Any date/time string using the supported formats.

        :param default:
            The default datetime object, if this is a datetime object and not
            ``None``, elements specified in ``timestr`` replace elements in the
            default object.

        :param ignoretz:
            If set ``True``, time zones in parsed strings are ignored and a
            naive :class:`datetime.datetime` object is returned.

        :param tzinfos:
            Additional time zone names / aliases which may be present in the
            string. This argument maps time zone names (and optionally offsets
            from those time zones) to time zones. This parameter can be a
            dictionary with timezone aliases mapping time zone names to time
            zones or a function taking two parameters (``tzname`` and
            ``tzoffset``) and returning a time zone.

            The timezones to which the names are mapped can be an integer
            offset from UTC in seconds or a :class:`tzinfo` object.

            .. doctest::
               :options: +NORMALIZE_WHITESPACE

                >>> from dateutil.parser import parse
                >>> from dateutil.tz import gettz
                >>> tzinfos = {"BRST": -7200, "CST": gettz("America/Chicago")}
                >>> parse("2012-01-19 17:21:00 BRST", tzinfos=tzinfos)
                datetime.datetime(2012, 1, 19, 17, 21, tzinfo=tzoffset(u'BRST', -7200))
                >>> parse("2012-01-19 17:21:00 CST", tzinfos=tzinfos)
                datetime.datetime(2012, 1, 19, 17, 21,
                                  tzinfo=tzfile('/usr/share/zoneinfo/America/Chicago'))

            This parameter is ignored if ``ignoretz`` is set.

        :param \\*\\*kwargs:
            Keyword arguments as passed to ``_parse()``.

        :return:
            Returns a :class:`datetime.datetime` object or, if the
            ``fuzzy_with_tokens`` option is ``True``, returns a tuple, the
            first element being a :class:`datetime.datetime` object, the second
            a tuple containing the fuzzy tokens.

        :raises ParserError:
            Raised for invalid or unknown string format, if the provided
            :class:`tzinfo` is not in a valid format, or if an invalid date
            would be created.

        :raises TypeError:
            Raised for non-string or character stream input.

        :raises OverflowError:
            Raised if the parsed date exceeds the largest valid C integer on
            your system.
        """

        if default is None:
            default = datetime.datetime.now().replace(hour=0, minute=0,
                                                      second=0, microsecond=0)

        res, skipped_tokens = self._parse(timestr, **kwargs)

        if res is None:
            raise ParserError("Unknown string format: %s", timestr)

        if len(res) == 0:
            raise ParserError("String does not contain a date: %s", timestr)

        try:
            ret = self._build_naive(res, default)
        except ValueError as e:
            six.raise_from(ParserError(str(e) + ": %s", timestr), e)

        if not ignoretz:
            ret = self._build_tzaware(ret, res, tzinfos)

        if kwargs.get('fuzzy_with_tokens', False):
            return ret, skipped_tokens
        else:
            return ret

    class _result(_resultbase):
        __slots__ = ["year", "month", "day", "weekday",
                     "hour", "minute", "second", "microsecond",
                     "tzname", "tzoffset", "ampm","any_unused_tokens"]

    def _parse(self, timestr, dayfirst=None, yearfirst=None, fuzzy=False,
               fuzzy_with_tokens=False):
        """
        Private method which performs the heavy lifting of parsing, called from
        ``parse()``, which passes on its ``kwargs`` to this function.

        :param timestr:
            The string to parse.

        :param dayfirst:
            Whether to interpret the first value in an ambiguous 3-integer date
            (e.g. 01/05/09) as the day (``True``) or month (``False``). If
            ``yearfirst`` is set to ``True``, this distinguishes between YDM
            and YMD. If set to ``None``, this value is retrieved from the
            current :class:`parserinfo` object (which itself defaults to
            ``False``).

        :param yearfirst:
            Whether to interpret the first value in an ambiguous 3-integer date
            (e.g. 01/05/09) as the year. If ``True``, the first number is taken
            to be the year, otherwise the last number is taken to be the year.
            If this is set to ``None``, the value is retrieved from the current
            :class:`parserinfo` object (which itself defaults to ``False``).

        :param fuzzy:
            Whether to allow fuzzy parsing, allowing for string like "Today is
            January 1, 2047 at 8:21:00AM".

        :param fuzzy_with_tokens:
            If ``True``, ``fuzzy`` is automatically set to True, and the parser
            will return a tuple where the first element is the parsed
            :class:`datetime.datetime` datetimestamp and the second element is
            a tuple containing the portions of the string which were ignored:

            .. doctest::

                >>> from dateutil.parser import parse
                >>> parse("Today is January 1, 2047 at 8:21:00AM", fuzzy_with_tokens=True)
                (datetime.datetime(2047, 1, 1, 8, 21), (u'Today is ', u' ', u'at '))

        """
        if fuzzy_with_tokens:
            fuzzy = True

        info = self.info

        if dayfirst is None:
            dayfirst = info.dayfirst

        if yearfirst is None:
            yearfirst = info.yearfirst

        res = self._result()
        l = _timelex.split(timestr)         # Splits the timestr into tokens

        skipped_idxs = []

        # year/month/day list
        ymd = _ymd()

        len_l = len(l)
        i = 0
        try:
            while i < len_l:

                # Check if it's a number
                value_repr = l[i]
                try:
                    value = float(value_repr)
                except ValueError:
                    value = None

                if value is not None:
                    # Numeric token
                    i = self._parse_numeric_token(l, i, info, ymd, res, fuzzy)

                # Check weekday
                elif info.weekday(l[i]) is not None:
                    value = info.weekday(l[i])
                    res.weekday = value

                # Check month name
                elif info.month(l[i]) is not None:
                    value = info.month(l[i])
                    ymd.append(value, 'M')

                    if i + 1 < len_l:
                        if l[i + 1] in ('-', '/'):
                            # Jan-01[-99]
                            sep = l[i + 1]
                            ymd.append(l[i + 2])

                            if i + 3 < len_l and l[i + 3] == sep:
                                # Jan-01-99
                                ymd.append(l[i + 4])
                                i += 2

                            i += 2

                        elif (i + 4 < len_l and l[i + 1] == l[i + 3] == ' ' and
                              info.pertain(l[i + 2])):
                            # Jan of 01
                            # In this case, 01 is clearly year
                            if l[i + 4].isdigit():
                                # Convert it here to become unambiguous
                                value = int(l[i + 4])
                                year = str(info.convertyear(value))
                                ymd.append(year, 'Y')
                            else:
                                # Wrong guess
                                pass
                                # TODO: not hit in tests
                            i += 4

                # Check am/pm
                elif info.ampm(l[i]) is not None:
                    value = info.ampm(l[i])
                    val_is_ampm = self._ampm_valid(res.hour, res.ampm, fuzzy)

                    if val_is_ampm:
                        res.hour = self._adjust_ampm(res.hour, value)
                        res.ampm = value

                    elif fuzzy:
                        skipped_idxs.append(i)

                # Check for a timezone name
                elif self._could_be_tzname(res.hour, res.tzname, res.tzoffset, l[i]):
                    res.tzname = l[i]
                    res.tzoffset = info.tzoffset(res.tzname)

                    # Check for something like GMT+3, or BRST+3. Notice
                    # that it doesn't mean "I am 3 hours after GMT", but
                    # "my time +3 is GMT". If found, we reverse the
                    # logic so that timezone parsing code will get it
                    # right.
                    if i + 1 < len_l and l[i + 1] in ('+', '-'):
                        l[i + 1] = ('+', '-')[l[i + 1] == '+']
                        res.tzoffset = None
                        if info.utczone(res.tzname):
                            # With something like GMT+3, the timezone
                            # is *not* GMT.
                            res.tzname = None

                # Check for a numbered timezone
                elif res.hour is not None and l[i] in ('+', '-'):
                    signal = (-1, 1)[l[i] == '+']
                    len_li = len(l[i + 1])

                    # TODO: check that l[i + 1] is integer?
                    if len_li == 4:
                        # -0300
                        hour_offset = int(l[i + 1][:2])
                        min_offset = int(l[i + 1][2:])
                    elif i + 2 < len_l and l[i + 2] == ':':
                        # -03:00
                        hour_offset = int(l[i + 1])
                        min_offset = int(l[i + 3])  # TODO: Check that l[i+3] is minute-like?
                        i += 2
                    elif len_li <= 2:
                        # -[0]3
                        hour_offset = int(l[i + 1][:2])
                        min_offset = 0
                    else:
                        raise ValueError(timestr)

                    res.tzoffset = signal * (hour_offset * 3600 + min_offset * 60)

                    # Look for a timezone name between parenthesis
                    if (i + 5 < len_l and
                            info.jump(l[i + 2]) and l[i + 3] == '(' and
                            l[i + 5] == ')' and
                            3 <= len(l[i + 4]) and
                            self._could_be_tzname(res.hour, res.tzname,
                                                  None, l[i + 4])):
                        # -0300 (BRST)
                        res.tzname = l[i + 4]
                        i += 4

                    i += 1

                # Check jumps
                elif not (info.jump(l[i]) or fuzzy):
                    raise ValueError(timestr)

                else:
                    skipped_idxs.append(i)
                i += 1

            # Process year/month/day
            year, month, day = ymd.resolve_ymd(yearfirst, dayfirst)

            res.century_specified = ymd.century_specified
            res.year = year
            res.month = month
            res.day = day

        except (IndexError, ValueError):
            return None, None

        if not info.validate(res):
            return None, None

        if fuzzy_with_tokens:
            skipped_tokens = self._recombine_skipped(l, skipped_idxs)
            return res, tuple(skipped_tokens)
        else:
            return res, None

    def _parse_numeric_token(self, tokens, idx, info, ymd, res, fuzzy):
        # Token is a number
        value_repr = tokens[idx]
        try:
            value = self._to_decimal(value_repr)
        except Exception as e:
            six.raise_from(ValueError('Unknown numeric token'), e)

        len_li = len(value_repr)

        len_l = len(tokens)

        if (len(ymd) == 3 and len_li in (2, 4) and
            res.hour is None and
            (idx + 1 >= len_l or
             (tokens[idx + 1] != ':' and
              info.hms(tokens[idx + 1]) is None))):
            # 19990101T23[59]
            s = tokens[idx]
            res.hour = int(s[:2])

            if len_li == 4:
                res.minute = int(s[2:])

        elif len_li == 6 or (len_li > 6 and tokens[idx].find('.') == 6):
            # YYMMDD or HHMMSS[.ss]
            s = tokens[idx]

            if not ymd and '.' not in tokens[idx]:
                ymd.append(s[:2])
                ymd.append(s[2:4])
                ymd.append(s[4:])
            else:
                # 19990101T235959[.59]

                # TODO: Check if res attributes already set.
                res.hour = int(s[:2])
                res.minute = int(s[2:4])
                res.second, res.microsecond = self._parsems(s[4:])

        elif len_li in (8, 12, 14):
            # YYYYMMDD
            s = tokens[idx]
            ymd.append(s[:4], 'Y')
            ymd.append(s[4:6])
            ymd.append(s[6:8])

            if len_li > 8:
                res.hour = int(s[8:10])
                res.minute = int(s[10:12])

                if len_li > 12:
                    res.second = int(s[12:])

        elif self._find_hms_idx(idx, tokens, info, allow_jump=True) is not None:
            # HH[ ]h or MM[ ]m or SS[.ss][ ]s
            hms_idx = self._find_hms_idx(idx, tokens, info, allow_jump=True)
            (idx, hms) = self._parse_hms(idx, tokens, info, hms_idx)
            if hms is not None:
                # TODO: checking that hour/minute/second are not
                # already set?
                self._assign_hms(res, value_repr, hms)

        elif idx + 2 < len_l and tokens[idx + 1] == ':':
            # HH:MM[:SS[.ss]]
            res.hour = int(value)
            value = self._to_decimal(tokens[idx + 2])  # TODO: try/except for this?
            (res.minute, res.second) = self._parse_min_sec(value)

            if idx + 4 < len_l and tokens[idx + 3] == ':':
                res.second, res.microsecond = self._parsems(tokens[idx + 4])

                idx += 2

            idx += 2

        elif idx + 1 < len_l and tokens[idx + 1] in ('-', '/', '.'):
            sep = tokens[idx + 1]
            ymd.append(value_repr)

            if idx + 2 < len_l and not info.jump(tokens[idx + 2]):
                if tokens[idx + 2].isdigit():
                    # 01-01[-01]
                    ymd.append(tokens[idx + 2])
                else:
                    # 01-Jan[-01]
                    value = info.month(tokens[idx + 2])

                    if value is not None:
                        ymd.append(value, 'M')
                    else:
                        raise ValueError()

                if idx + 3 < len_l and tokens[idx + 3] == sep:
                    # We have three members
                    value = info.month(tokens[idx + 4])

                    if value is not None:
                        ymd.append(value, 'M')
                    else:
                        ymd.append(tokens[idx + 4])
                    idx += 2

                idx += 1
            idx += 1

        elif idx + 1 >= len_l or info.jump(tokens[idx + 1]):
            if idx + 2 < len_l and info.ampm(tokens[idx + 2]) is not None:
                # 12 am
                hour = int(value)
                res.hour = self._adjust_ampm(hour, info.ampm(tokens[idx + 2]))
                idx += 1
            else:
                # Year, month or day
                ymd.append(value)
            idx += 1

        elif info.ampm(tokens[idx + 1]) is not None and (0 <= value < 24):
            # 12am
            hour = int(value)
            res.hour = self._adjust_ampm(hour, info.ampm(tokens[idx + 1]))
            idx += 1

        elif ymd.could_be_day(value):
            ymd.append(value)

        elif not fuzzy:
            raise ValueError()

        return idx

    def _find_hms_idx(self, idx, tokens, info, allow_jump):
        len_l = len(tokens)

        if idx+1 < len_l and info.hms(tokens[idx+1]) is not None:
            # There is an "h", "m", or "s" label following this token.  We take
            # assign the upcoming label to the current token.
            # e.g. the "12" in 12h"
            hms_idx = idx + 1

        elif (allow_jump and idx+2 < len_l and tokens[idx+1] == ' ' and
              info.hms(tokens[idx+2]) is not None):
            # There is a space and then an "h", "m", or "s" label.
            # e.g. the "12" in "12 h"
            hms_idx = idx + 2

        elif idx > 0 and info.hms(tokens[idx-1]) is not None:
            # There is a "h", "m", or "s" preceding this token.  Since neither
            # of the previous cases was hit, there is no label following this
            # token, so we use the previous label.
            # e.g. the "04" in "12h04"
            hms_idx = idx-1

        elif (1 < idx == len_l-1 and tokens[idx-1] == ' ' and
              info.hms(tokens[idx-2]) is not None):
            # If we are looking at the final token, we allow for a
            # backward-looking check to skip over a space.
            # TODO: Are we sure this is the right condition here?
            hms_idx = idx - 2

        else:
            hms_idx = None

        return hms_idx

    def _assign_hms(self, res, value_repr, hms):
        # See GH issue #427, fixing float rounding
        value = self._to_decimal(value_repr)

        if hms == 0:
            # Hour
            res.hour = int(value)
            if value % 1:
                res.minute = int(60*(value % 1))

        elif hms == 1:
            (res.minute, res.second) = self._parse_min_sec(value)

        elif hms == 2:
            (res.second, res.microsecond) = self._parsems(value_repr)

    def _could_be_tzname(self, hour, tzname, tzoffset, token):
        return (hour is not None and
                tzname is None and
                tzoffset is None and
                len(token) <= 5 and
                (all(x in string.ascii_uppercase for x in token)
                 or token in self.info.UTCZONE))

    def _ampm_valid(self, hour, ampm, fuzzy):
        """
        For fuzzy parsing, 'a' or 'am' (both valid English words)
        may erroneously trigger the AM/PM flag. Deal with that
        here.
        """
        val_is_ampm = True

        # If there's already an AM/PM flag, this one isn't one.
        if fuzzy and ampm is not None:
            val_is_ampm = False

        # If AM/PM is found and hour is not, raise a ValueError
        if hour is None:
            if fuzzy:
                val_is_ampm = False
            else:
                raise ValueError('No hour specified with AM or PM flag.')
        elif not 0 <= hour <= 12:
            # If AM/PM is found, it's a 12 hour clock, so raise
            # an error for invalid range
            if fuzzy:
                val_is_ampm = False
            else:
                raise ValueError('Invalid hour specified for 12-hour clock.')

        return val_is_ampm

    def _adjust_ampm(self, hour, ampm):
        if hour < 12 and ampm == 1:
            hour += 12
        elif hour == 12 and ampm == 0:
            hour = 0
        return hour

    def _parse_min_sec(self, value):
        # TODO: Every usage of this function sets res.second to the return
        # value. Are there any cases where second will be returned as None and
        # we *don't* want to set res.second = None?
        minute = int(value)
        second = None

        sec_remainder = value % 1
        if sec_remainder:
            second = int(60 * sec_remainder)
        return (minute, second)

    def _parse_hms(self, idx, tokens, info, hms_idx):
        # TODO: Is this going to admit a lot of false-positives for when we
        # just happen to have digits and "h", "m" or "s" characters in non-date
        # text?  I guess hex hashes won't have that problem, but there's plenty
        # of random junk out there.
        if hms_idx is None:
            hms = None
            new_idx = idx
        elif hms_idx > idx:
            hms = info.hms(tokens[hms_idx])
            new_idx = hms_idx
        else:
            # Looking backwards, increment one.
            hms = info.hms(tokens[hms_idx]) + 1
            new_idx = idx

        return (new_idx, hms)

    # ------------------------------------------------------------------
    # Handling for individual tokens.  These are kept as methods instead
    #  of functions for the sake of customizability via subclassing.

    def _parsems(self, value):
        """Parse a I[.F] seconds value into (seconds, microseconds)."""
        if "." not in value:
            return int(value), 0
        else:
            i, f = value.split(".")
            return int(i), int(f.ljust(6, "0")[:6])

    def _to_decimal(self, val):
        try:
            decimal_value = Decimal(val)
            # See GH 662, edge case, infinite value should not be converted
            #  via `_to_decimal`
            if not decimal_value.is_finite():
                raise ValueError("Converted decimal value is infinite or NaN")
        except Exception as e:
            msg = "Could not convert %s to decimal" % val
            six.raise_from(ValueError(msg), e)
        else:
            return decimal_value

    # ------------------------------------------------------------------
    # Post-Parsing construction of datetime output.  These are kept as
    #  methods instead of functions for the sake of customizability via
    #  subclassing.

    def _build_tzinfo(self, tzinfos, tzname, tzoffset):
        if callable(tzinfos):
            tzdata = tzinfos(tzname, tzoffset)
        else:
            tzdata = tzinfos.get(tzname)
        # handle case where tzinfo is paased an options that returns None
        # eg tzinfos = {'BRST' : None}
        if isinstance(tzdata, datetime.tzinfo) or tzdata is None:
            tzinfo = tzdata
        elif isinstance(tzdata, text_type):
            tzinfo = tz.tzstr(tzdata)
        elif isinstance(tzdata, integer_types):
            tzinfo = tz.tzoffset(tzname, tzdata)
        else:
            raise TypeError("Offset must be tzinfo subclass, tz string, "
                            "or int offset.")
        return tzinfo

    def _build_tzaware(self, naive, res, tzinfos):
        if (callable(tzinfos) or (tzinfos and res.tzname in tzinfos)):
            tzinfo = self._build_tzinfo(tzinfos, res.tzname, res.tzoffset)
            aware = naive.replace(tzinfo=tzinfo)
            aware = self._assign_tzname(aware, res.tzname)

        elif res.tzname and res.tzname in time.tzname:
            aware = naive.replace(tzinfo=tz.tzlocal())

            # Handle ambiguous local datetime
            aware = self._assign_tzname(aware, res.tzname)

            # This is mostly relevant for winter GMT zones parsed in the UK
            if (aware.tzname() != res.tzname and
                    res.tzname in self.info.UTCZONE):
                aware = aware.replace(tzinfo=tz.UTC)

        elif res.tzoffset == 0:
            aware = naive.replace(tzinfo=tz.UTC)

        elif res.tzoffset:
            aware = naive.replace(tzinfo=tz.tzoffset(res.tzname, res.tzoffset))

        elif not res.tzname and not res.tzoffset:
            # i.e. no timezone information was found.
            aware = naive

        elif res.tzname:
            # tz-like string was parsed but we don't know what to do
            # with it
            warnings.warn("tzname {tzname} identified but not understood.  "
                          "Pass `tzinfos` argument in order to correctly "
                          "return a timezone-aware datetime.  In a future "
                          "version, this will raise an "
                          "exception.".format(tzname=res.tzname),
                          category=UnknownTimezoneWarning)
            aware = naive

        return aware

    def _build_naive(self, res, default):
        repl = {}
        for attr in ("year", "month", "day", "hour",
                     "minute", "second", "microsecond"):
            value = getattr(res, attr)
            if value is not None:
                repl[attr] = value

        if 'day' not in repl:
            # If the default day exceeds the last day of the month, fall back
            # to the end of the month.
            cyear = default.year if res.year is None else res.year
            cmonth = default.month if res.month is None else res.month
            cday = default.day if res.day is None else res.day

            if cday > monthrange(cyear, cmonth)[1]:
                repl['day'] = monthrange(cyear, cmonth)[1]

        naive = default.replace(**repl)

        if res.weekday is not None and not res.day:
            naive = naive + relativedelta.relativedelta(weekday=res.weekday)

        return naive

    def _assign_tzname(self, dt, tzname):
        if dt.tzname() != tzname:
            new_dt = tz.enfold(dt, fold=1)
            if new_dt.tzname() == tzname:
                return new_dt

        return dt

    def _recombine_skipped(self, tokens, skipped_idxs):
        """
        >>> tokens = ["foo", " ", "bar", " ", "19June2000", "baz"]
        >>> skipped_idxs = [0, 1, 2, 5]
        >>> _recombine_skipped(tokens, skipped_idxs)
        ["foo bar", "baz"]
        """
        skipped_tokens = []
        for i, idx in enumerate(sorted(skipped_idxs)):
            if i > 0 and idx - 1 == skipped_idxs[i - 1]:
                skipped_tokens[-1] = skipped_tokens[-1] + tokens[idx]
            else:
                skipped_tokens.append(tokens[idx])

        return skipped_tokens


DEFAULTPARSER = parser()


def parse(timestr, parserinfo=None, **kwargs):
    """

    Parse a string in one of the supported formats, using the
    ``parserinfo`` parameters.

    :param timestr:
        A string containing a date/time stamp.

    :param parserinfo:
        A :class:`parserinfo` object containing parameters for the parser.
        If ``None``, the default arguments to the :class:`parserinfo`
        constructor are used.

    The ``**kwargs`` parameter takes the following keyword arguments:

    :param default:
        The default datetime object, if this is a datetime object and not
        ``None``, elements specified in ``timestr`` replace elements in the
        default object.

    :param ignoretz:
        If set ``True``, time zones in parsed strings are ignored and a naive
        :class:`datetime` object is returned.

    :param tzinfos:
        Additional time zone names / aliases which may be present in the
        string. This argument maps time zone names (and optionally offsets
        from those time zones) to time zones. This parameter can be a
        dictionary with timezone aliases mapping time zone names to time
        zones or a function taking two parameters (``tzname`` and
        ``tzoffset``) and returning a time zone.

        The timezones to which the names are mapped can be an integer
        offset from UTC in seconds or a :class:`tzinfo` object.

        .. doctest::
           :options: +NORMALIZE_WHITESPACE

            >>> from dateutil.parser import parse
            >>> from dateutil.tz import gettz
            >>> tzinfos = {"BRST": -7200, "CST": gettz("America/Chicago")}
            >>> parse("2012-01-19 17:21:00 BRST", tzinfos=tzinfos)
            datetime.datetime(2012, 1, 19, 17, 21, tzinfo=tzoffset(u'BRST', -7200))
            >>> parse("2012-01-19 17:21:00 CST", tzinfos=tzinfos)
            datetime.datetime(2012, 1, 19, 17, 21,
                              tzinfo=tzfile('/usr/share/zoneinfo/America/Chicago'))

        This parameter is ignored if ``ignoretz`` is set.

    :param dayfirst:
        Whether to interpret the first value in an ambiguous 3-integer date
        (e.g. 01/05/09) as the day (``True``) or month (``False``). If
        ``yearfirst`` is set to ``True``, this distinguishes between YDM and
        YMD. If set to ``None``, this value is retrieved from the current
        :class:`parserinfo` object (which itself defaults to ``False``).

    :param yearfirst:
        Whether to interpret the first value in an ambiguous 3-integer date
        (e.g. 01/05/09) as the year. If ``True``, the first number is taken to
        be the year, otherwise the last number is taken to be the year. If
        this is set to ``None``, the value is retrieved from the current
        :class:`parserinfo` object (which itself defaults to ``False``).

    :param fuzzy:
        Whether to allow fuzzy parsing, allowing for string like "Today is
        January 1, 2047 at 8:21:00AM".

    :param fuzzy_with_tokens:
        If ``True``, ``fuzzy`` is automatically set to True, and the parser
        will return a tuple where the first element is the parsed
        :class:`datetime.datetime` datetimestamp and the second element is
        a tuple containing the portions of the string which were ignored:

        .. doctest::

            >>> from dateutil.parser import parse
            >>> parse("Today is January 1, 2047 at 8:21:00AM", fuzzy_with_tokens=True)
            (datetime.datetime(2047, 1, 1, 8, 21), (u'Today is ', u' ', u'at '))

    :return:
        Returns a :class:`datetime.datetime` object or, if the
        ``fuzzy_with_tokens`` option is ``True``, returns a tuple, the
        first element being a :class:`datetime.datetime` object, the second
        a tuple containing the fuzzy tokens.

    :raises ParserError:
        Raised for invalid or unknown string formats, if the provided
        :class:`tzinfo` is not in a valid format, or if an invalid date would
        be created.

    :raises OverflowError:
        Raised if the parsed date exceeds the largest valid C integer on
        your system.
    """
    if parserinfo:
        return parser(parserinfo).parse(timestr, **kwargs)
    else:
        return DEFAULTPARSER.parse(timestr, **kwargs)


class _tzparser(object):

    class _result(_resultbase):

        __slots__ = ["stdabbr", "stdoffset", "dstabbr", "dstoffset",
                     "start", "end"]

        class _attr(_resultbase):
            __slots__ = ["month", "week", "weekday",
                         "yday", "jyday", "day", "time"]

        def __repr__(self):
            return self._repr("")

        def __init__(self):
            _resultbase.__init__(self)
            self.start = self._attr()
            self.end = self._attr()

    def parse(self, tzstr):
        res = self._result()
        l = [x for x in re.split(r'([,:.]|[a-zA-Z]+|[0-9]+)',tzstr) if x]
        used_idxs = list()
        try:

            len_l = len(l)

            i = 0
            while i < len_l:
                # BRST+3[BRDT[+2]]
                j = i
                while j < len_l and not [x for x in l[j]
                                         if x in "0123456789:,-+"]:
                    j += 1
                if j != i:
                    if not res.stdabbr:
                        offattr = "stdoffset"
                        res.stdabbr = "".join(l[i:j])
                    else:
                        offattr = "dstoffset"
                        res.dstabbr = "".join(l[i:j])

                    for ii in range(j):
                        used_idxs.append(ii)
                    i = j
                    if (i < len_l and (l[i] in ('+', '-') or l[i][0] in
                                       "0123456789")):
                        if l[i] in ('+', '-'):
                            # Yes, that's right.  See the TZ variable
                            # documentation.
                            signal = (1, -1)[l[i] == '+']
                            used_idxs.append(i)
                            i += 1
                        else:
                            signal = -1
                        len_li = len(l[i])
                        if len_li == 4:
                            # -0300
                            setattr(res, offattr, (int(l[i][:2]) * 3600 +
                                                   int(l[i][2:]) * 60) * signal)
                        elif i + 1 < len_l and l[i + 1] == ':':
                            # -03:00
                            setattr(res, offattr,
                                    (int(l[i]) * 3600 +
                                     int(l[i + 2]) * 60) * signal)
                            used_idxs.append(i)
                            i += 2
                        elif len_li <= 2:
                            # -[0]3
                            setattr(res, offattr,
                                    int(l[i][:2]) * 3600 * signal)
                        else:
                            return None
                        used_idxs.append(i)
                        i += 1
                    if res.dstabbr:
                        break
                else:
                    break


            if i < len_l:
                for j in range(i, len_l):
                    if l[j] == ';':
                        l[j] = ','

                assert l[i] == ','

                i += 1

            if i >= len_l:
                pass
            elif (8 <= l.count(',') <= 9 and
                  not [y for x in l[i:] if x != ','
                       for y in x if y not in "0123456789+-"]):
                # GMT0BST,3,0,30,3600,10,0,26,7200[,3600]
                for x in (res.start, res.end):
                    x.month = int(l[i])
                    used_idxs.append(i)
                    i += 2
                    if l[i] == '-':
                        value = int(l[i + 1]) * -1
                        used_idxs.append(i)
                        i += 1
                    else:
                        value = int(l[i])
                    used_idxs.append(i)
                    i += 2
                    if value:
                        x.week = value
                        x.weekday = (int(l[i]) - 1) % 7
                    else:
                        x.day = int(l[i])
                    used_idxs.append(i)
                    i += 2
                    x.time = int(l[i])
                    used_idxs.append(i)
                    i += 2
                if i < len_l:
                    if l[i] in ('-', '+'):
                        signal = (-1, 1)[l[i] == "+"]
                        used_idxs.append(i)
                        i += 1
                    else:
                        signal = 1
                    used_idxs.append(i)
                    res.dstoffset = (res.stdoffset + int(l[i]) * signal)

                # This was a made-up format that is not in normal use
                warn(('Parsed time zone "%s"' % tzstr) +
                     'is in a non-standard dateutil-specific format, which ' +
                     'is now deprecated; support for parsing this format ' +
                     'will be removed in future versions. It is recommended ' +
                     'that you switch to a standard format like the GNU ' +
                     'TZ variable format.', tz.DeprecatedTzFormatWarning)
            elif (l.count(',') == 2 and l[i:].count('/') <= 2 and
                  not [y for x in l[i:] if x not in (',', '/', 'J', 'M',
                                                     '.', '-', ':')
                       for y in x if y not in "0123456789"]):
                for x in (res.start, res.end):
                    if l[i] == 'J':
                        # non-leap year day (1 based)
                        used_idxs.append(i)
                        i += 1
                        x.jyday = int(l[i])
                    elif l[i] == 'M':
                        # month[-.]week[-.]weekday
                        used_idxs.append(i)
                        i += 1
                        x.month = int(l[i])
                        used_idxs.append(i)
                        i += 1
                        assert l[i] in ('-', '.')
                        used_idxs.append(i)
                        i += 1
                        x.week = int(l[i])
                        if x.week == 5:
                            x.week = -1
                        used_idxs.append(i)
                        i += 1
                        assert l[i] in ('-', '.')
                        used_idxs.append(i)
                        i += 1
                        x.weekday = (int(l[i]) - 1) % 7
                    else:
                        # year day (zero based)
                        x.yday = int(l[i]) + 1

                    used_idxs.append(i)
                    i += 1

                    if i < len_l and l[i] == '/':
                        used_idxs.append(i)
                        i += 1
                        # start time
                        len_li = len(l[i])
                        if len_li == 4:
                            # -0300
                            x.time = (int(l[i][:2]) * 3600 +
                                      int(l[i][2:]) * 60)
                        elif i + 1 < len_l and l[i + 1] == ':':
                            # -03:00
                            x.time = int(l[i]) * 3600 + int(l[i + 2]) * 60
                            used_idxs.append(i)
                            i += 2
                            if i + 1 < len_l and l[i + 1] == ':':
                                used_idxs.append(i)
                                i += 2
                                x.time += int(l[i])
                        elif len_li <= 2:
                            # -[0]3
                            x.time = (int(l[i][:2]) * 3600)
                        else:
                            return None
                        used_idxs.append(i)
                        i += 1

                    assert i == len_l or l[i] == ','

                    i += 1

                assert i >= len_l

        except (IndexError, ValueError, AssertionError):
            return None

        unused_idxs = set(range(len_l)).difference(used_idxs)
        res.any_unused_tokens = not {l[n] for n in unused_idxs}.issubset({",",":"})
        return res


DEFAULTTZPARSER = _tzparser()


def _parsetz(tzstr):
    return DEFAULTTZPARSER.parse(tzstr)


class ParserError(ValueError):
    """Exception subclass used for any failure to parse a datetime string.

    This is a subclass of :py:exc:`ValueError`, and should be raised any time
    earlier versions of ``dateutil`` would have raised ``ValueError``.

    .. versionadded:: 2.8.1
    """
    def __str__(self):
        try:
            return self.args[0] % self.args[1:]
        except (TypeError, IndexError):
            return super(ParserError, self).__str__()

    def __repr__(self):
        args = ", ".join("'%s'" % arg for arg in self.args)
        return "%s(%s)" % (self.__class__.__name__, args)


class UnknownTimezoneWarning(RuntimeWarning):
    """Raised when the parser finds a timezone it cannot parse into a tzinfo.

    .. versionadded:: 2.7.0
    """
# vim:ts=4:sw=4:et

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\asyncio\scoping.py ===
# ext/asyncio/scoping.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from .session import _AS
from .session import async_sessionmaker
from .session import AsyncSession
from ... import exc as sa_exc
from ... import util
from ...orm.session import Session
from ...util import create_proxy_methods
from ...util import ScopedRegistry
from ...util import warn
from ...util import warn_deprecated

if TYPE_CHECKING:
    from .engine import AsyncConnection
    from .result import AsyncResult
    from .result import AsyncScalarResult
    from .session import AsyncSessionTransaction
    from ...engine import Connection
    from ...engine import CursorResult
    from ...engine import Engine
    from ...engine import Result
    from ...engine import Row
    from ...engine import RowMapping
    from ...engine.interfaces import _CoreAnyExecuteParams
    from ...engine.interfaces import CoreExecuteOptionsParameter
    from ...engine.result import ScalarResult
    from ...orm._typing import _IdentityKeyType
    from ...orm._typing import _O
    from ...orm._typing import OrmExecuteOptionsParameter
    from ...orm.interfaces import ORMOption
    from ...orm.session import _BindArguments
    from ...orm.session import _EntityBindKey
    from ...orm.session import _PKIdentityArgument
    from ...orm.session import _SessionBind
    from ...sql.base import Executable
    from ...sql.dml import UpdateBase
    from ...sql.elements import ClauseElement
    from ...sql.selectable import ForUpdateParameter
    from ...sql.selectable import TypedReturnsRows

_T = TypeVar("_T", bound=Any)


@create_proxy_methods(
    AsyncSession,
    ":class:`_asyncio.AsyncSession`",
    ":class:`_asyncio.scoping.async_scoped_session`",
    classmethods=["close_all", "object_session", "identity_key"],
    methods=[
        "__contains__",
        "__iter__",
        "aclose",
        "add",
        "add_all",
        "begin",
        "begin_nested",
        "close",
        "reset",
        "commit",
        "connection",
        "delete",
        "execute",
        "expire",
        "expire_all",
        "expunge",
        "expunge_all",
        "flush",
        "get_bind",
        "is_modified",
        "invalidate",
        "merge",
        "refresh",
        "rollback",
        "scalar",
        "scalars",
        "get",
        "get_one",
        "stream",
        "stream_scalars",
    ],
    attributes=[
        "bind",
        "dirty",
        "deleted",
        "new",
        "identity_map",
        "is_active",
        "autoflush",
        "no_autoflush",
        "info",
    ],
    use_intermediate_variable=["get"],
)
class async_scoped_session(Generic[_AS]):
    """Provides scoped management of :class:`.AsyncSession` objects.

    See the section :ref:`asyncio_scoped_session` for usage details.

    .. versionadded:: 1.4.19


    """

    _support_async = True

    session_factory: async_sessionmaker[_AS]
    """The `session_factory` provided to `__init__` is stored in this
    attribute and may be accessed at a later time.  This can be useful when
    a new non-scoped :class:`.AsyncSession` is needed."""

    registry: ScopedRegistry[_AS]

    def __init__(
        self,
        session_factory: async_sessionmaker[_AS],
        scopefunc: Callable[[], Any],
    ):
        """Construct a new :class:`_asyncio.async_scoped_session`.

        :param session_factory: a factory to create new :class:`_asyncio.AsyncSession`
         instances. This is usually, but not necessarily, an instance
         of :class:`_asyncio.async_sessionmaker`.

        :param scopefunc: function which defines
         the current scope.   A function such as ``asyncio.current_task``
         may be useful here.

        """  # noqa: E501

        self.session_factory = session_factory
        self.registry = ScopedRegistry(session_factory, scopefunc)

    @property
    def _proxied(self) -> _AS:
        return self.registry()

    def __call__(self, **kw: Any) -> _AS:
        r"""Return the current :class:`.AsyncSession`, creating it
        using the :attr:`.scoped_session.session_factory` if not present.

        :param \**kw: Keyword arguments will be passed to the
         :attr:`.scoped_session.session_factory` callable, if an existing
         :class:`.AsyncSession` is not present.  If the
         :class:`.AsyncSession` is present
         and keyword arguments have been passed,
         :exc:`~sqlalchemy.exc.InvalidRequestError` is raised.

        """
        if kw:
            if self.registry.has():
                raise sa_exc.InvalidRequestError(
                    "Scoped session is already present; "
                    "no new arguments may be specified."
                )
            else:
                sess = self.session_factory(**kw)
                self.registry.set(sess)
        else:
            sess = self.registry()
        if not self._support_async and sess._is_asyncio:
            warn_deprecated(
                "Using `scoped_session` with asyncio is deprecated and "
                "will raise an error in a future version. "
                "Please use `async_scoped_session` instead.",
                "1.4.23",
            )
        return sess

    def configure(self, **kwargs: Any) -> None:
        """reconfigure the :class:`.sessionmaker` used by this
        :class:`.scoped_session`.

        See :meth:`.sessionmaker.configure`.

        """

        if self.registry.has():
            warn(
                "At least one scoped session is already present. "
                " configure() can not affect sessions that have "
                "already been created."
            )

        self.session_factory.configure(**kwargs)

    async def remove(self) -> None:
        """Dispose of the current :class:`.AsyncSession`, if present.

        Different from scoped_session's remove method, this method would use
        await to wait for the close method of AsyncSession.

        """

        if self.registry.has():
            await self.registry().close()
        self.registry.clear()

    # START PROXY METHODS async_scoped_session

    # code within this block is **programmatically,
    # statically generated** by tools/generate_proxy_methods.py

    def __contains__(self, instance: object) -> bool:
        r"""Return True if the instance is associated with this session.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        The instance may be pending or persistent within the Session for a
        result of True.



        """  # noqa: E501

        return self._proxied.__contains__(instance)

    def __iter__(self) -> Iterator[object]:
        r"""Iterate over all pending or persistent instances within this
        Session.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.



        """  # noqa: E501

        return self._proxied.__iter__()

    async def aclose(self) -> None:
        r"""A synonym for :meth:`_asyncio.AsyncSession.close`.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        The :meth:`_asyncio.AsyncSession.aclose` name is specifically
        to support the Python standard library ``@contextlib.aclosing``
        context manager function.

        .. versionadded:: 2.0.20


        """  # noqa: E501

        return await self._proxied.aclose()

    def add(self, instance: object, _warn: bool = True) -> None:
        r"""Place an object into this :class:`_orm.Session`.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        Objects that are in the :term:`transient` state when passed to the
        :meth:`_orm.Session.add` method will move to the
        :term:`pending` state, until the next flush, at which point they
        will move to the :term:`persistent` state.

        Objects that are in the :term:`detached` state when passed to the
        :meth:`_orm.Session.add` method will move to the :term:`persistent`
        state directly.

        If the transaction used by the :class:`_orm.Session` is rolled back,
        objects which were transient when they were passed to
        :meth:`_orm.Session.add` will be moved back to the
        :term:`transient` state, and will no longer be present within this
        :class:`_orm.Session`.

        .. seealso::

            :meth:`_orm.Session.add_all`

            :ref:`session_adding` - at :ref:`session_basics`



        """  # noqa: E501

        return self._proxied.add(instance, _warn=_warn)

    def add_all(self, instances: Iterable[object]) -> None:
        r"""Add the given collection of instances to this :class:`_orm.Session`.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        See the documentation for :meth:`_orm.Session.add` for a general
        behavioral description.

        .. seealso::

            :meth:`_orm.Session.add`

            :ref:`session_adding` - at :ref:`session_basics`



        """  # noqa: E501

        return self._proxied.add_all(instances)

    def begin(self) -> AsyncSessionTransaction:
        r"""Return an :class:`_asyncio.AsyncSessionTransaction` object.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        The underlying :class:`_orm.Session` will perform the
        "begin" action when the :class:`_asyncio.AsyncSessionTransaction`
        object is entered::

            async with async_session.begin():
                ...  # ORM transaction is begun

        Note that database IO will not normally occur when the session-level
        transaction is begun, as database transactions begin on an
        on-demand basis.  However, the begin block is async to accommodate
        for a :meth:`_orm.SessionEvents.after_transaction_create`
        event hook that may perform IO.

        For a general description of ORM begin, see
        :meth:`_orm.Session.begin`.


        """  # noqa: E501

        return self._proxied.begin()

    def begin_nested(self) -> AsyncSessionTransaction:
        r"""Return an :class:`_asyncio.AsyncSessionTransaction` object
        which will begin a "nested" transaction, e.g. SAVEPOINT.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        Behavior is the same as that of :meth:`_asyncio.AsyncSession.begin`.

        For a general description of ORM begin nested, see
        :meth:`_orm.Session.begin_nested`.

        .. seealso::

            :ref:`aiosqlite_serializable` - special workarounds required
            with the SQLite asyncio driver in order for SAVEPOINT to work
            correctly.


        """  # noqa: E501

        return self._proxied.begin_nested()

    async def close(self) -> None:
        r"""Close out the transactional resources and ORM objects used by this
        :class:`_asyncio.AsyncSession`.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. seealso::

            :meth:`_orm.Session.close` - main documentation for
            "close"

            :ref:`session_closing` - detail on the semantics of
            :meth:`_asyncio.AsyncSession.close` and
            :meth:`_asyncio.AsyncSession.reset`.


        """  # noqa: E501

        return await self._proxied.close()

    async def reset(self) -> None:
        r"""Close out the transactional resources and ORM objects used by this
        :class:`_orm.Session`, resetting the session to its initial state.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. versionadded:: 2.0.22

        .. seealso::

            :meth:`_orm.Session.reset` - main documentation for
            "reset"

            :ref:`session_closing` - detail on the semantics of
            :meth:`_asyncio.AsyncSession.close` and
            :meth:`_asyncio.AsyncSession.reset`.


        """  # noqa: E501

        return await self._proxied.reset()

    async def commit(self) -> None:
        r"""Commit the current transaction in progress.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. seealso::

            :meth:`_orm.Session.commit` - main documentation for
            "commit"

        """  # noqa: E501

        return await self._proxied.commit()

    async def connection(
        self,
        bind_arguments: Optional[_BindArguments] = None,
        execution_options: Optional[CoreExecuteOptionsParameter] = None,
        **kw: Any,
    ) -> AsyncConnection:
        r"""Return a :class:`_asyncio.AsyncConnection` object corresponding to
        this :class:`.Session` object's transactional state.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        This method may also be used to establish execution options for the
        database connection used by the current transaction.

        .. versionadded:: 1.4.24  Added \**kw arguments which are passed
           through to the underlying :meth:`_orm.Session.connection` method.

        .. seealso::

            :meth:`_orm.Session.connection` - main documentation for
            "connection"


        """  # noqa: E501

        return await self._proxied.connection(
            bind_arguments=bind_arguments,
            execution_options=execution_options,
            **kw,
        )

    async def delete(self, instance: object) -> None:
        r"""Mark an instance as deleted.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        The database delete operation occurs upon ``flush()``.

        As this operation may need to cascade along unloaded relationships,
        it is awaitable to allow for those queries to take place.

        .. seealso::

            :meth:`_orm.Session.delete` - main documentation for delete


        """  # noqa: E501

        return await self._proxied.delete(instance)

    @overload
    async def execute(
        self,
        statement: TypedReturnsRows[_T],
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        _parent_execute_state: Optional[Any] = None,
        _add_event: Optional[Any] = None,
    ) -> Result[_T]: ...

    @overload
    async def execute(
        self,
        statement: UpdateBase,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        _parent_execute_state: Optional[Any] = None,
        _add_event: Optional[Any] = None,
    ) -> CursorResult[Any]: ...

    @overload
    async def execute(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        _parent_execute_state: Optional[Any] = None,
        _add_event: Optional[Any] = None,
    ) -> Result[Any]: ...

    async def execute(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> Result[Any]:
        r"""Execute a statement and return a buffered
        :class:`_engine.Result` object.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. seealso::

            :meth:`_orm.Session.execute` - main documentation for execute


        """  # noqa: E501

        return await self._proxied.execute(
            statement,
            params=params,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
            **kw,
        )

    def expire(
        self, instance: object, attribute_names: Optional[Iterable[str]] = None
    ) -> None:
        r"""Expire the attributes on an instance.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        Marks the attributes of an instance as out of date. When an expired
        attribute is next accessed, a query will be issued to the
        :class:`.Session` object's current transactional context in order to
        load all expired attributes for the given instance.   Note that
        a highly isolated transaction will return the same values as were
        previously read in that same transaction, regardless of changes
        in database state outside of that transaction.

        To expire all objects in the :class:`.Session` simultaneously,
        use :meth:`Session.expire_all`.

        The :class:`.Session` object's default behavior is to
        expire all state whenever the :meth:`Session.rollback`
        or :meth:`Session.commit` methods are called, so that new
        state can be loaded for the new transaction.   For this reason,
        calling :meth:`Session.expire` only makes sense for the specific
        case that a non-ORM SQL statement was emitted in the current
        transaction.

        :param instance: The instance to be refreshed.
        :param attribute_names: optional list of string attribute names
          indicating a subset of attributes to be expired.

        .. seealso::

            :ref:`session_expire` - introductory material

            :meth:`.Session.expire`

            :meth:`.Session.refresh`

            :meth:`_orm.Query.populate_existing`



        """  # noqa: E501

        return self._proxied.expire(instance, attribute_names=attribute_names)

    def expire_all(self) -> None:
        r"""Expires all persistent instances within this Session.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        When any attributes on a persistent instance is next accessed,
        a query will be issued using the
        :class:`.Session` object's current transactional context in order to
        load all expired attributes for the given instance.   Note that
        a highly isolated transaction will return the same values as were
        previously read in that same transaction, regardless of changes
        in database state outside of that transaction.

        To expire individual objects and individual attributes
        on those objects, use :meth:`Session.expire`.

        The :class:`.Session` object's default behavior is to
        expire all state whenever the :meth:`Session.rollback`
        or :meth:`Session.commit` methods are called, so that new
        state can be loaded for the new transaction.   For this reason,
        calling :meth:`Session.expire_all` is not usually needed,
        assuming the transaction is isolated.

        .. seealso::

            :ref:`session_expire` - introductory material

            :meth:`.Session.expire`

            :meth:`.Session.refresh`

            :meth:`_orm.Query.populate_existing`



        """  # noqa: E501

        return self._proxied.expire_all()

    def expunge(self, instance: object) -> None:
        r"""Remove the `instance` from this ``Session``.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        This will free all internal references to the instance.  Cascading
        will be applied according to the *expunge* cascade rule.



        """  # noqa: E501

        return self._proxied.expunge(instance)

    def expunge_all(self) -> None:
        r"""Remove all object instances from this ``Session``.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        This is equivalent to calling ``expunge(obj)`` on all objects in this
        ``Session``.



        """  # noqa: E501

        return self._proxied.expunge_all()

    async def flush(self, objects: Optional[Sequence[Any]] = None) -> None:
        r"""Flush all the object changes to the database.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. seealso::

            :meth:`_orm.Session.flush` - main documentation for flush


        """  # noqa: E501

        return await self._proxied.flush(objects=objects)

    def get_bind(
        self,
        mapper: Optional[_EntityBindKey[_O]] = None,
        clause: Optional[ClauseElement] = None,
        bind: Optional[_SessionBind] = None,
        **kw: Any,
    ) -> Union[Engine, Connection]:
        r"""Return a "bind" to which the synchronous proxied :class:`_orm.Session`
        is bound.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        Unlike the :meth:`_orm.Session.get_bind` method, this method is
        currently **not** used by this :class:`.AsyncSession` in any way
        in order to resolve engines for requests.

        .. note::

            This method proxies directly to the :meth:`_orm.Session.get_bind`
            method, however is currently **not** useful as an override target,
            in contrast to that of the :meth:`_orm.Session.get_bind` method.
            The example below illustrates how to implement custom
            :meth:`_orm.Session.get_bind` schemes that work with
            :class:`.AsyncSession` and :class:`.AsyncEngine`.

        The pattern introduced at :ref:`session_custom_partitioning`
        illustrates how to apply a custom bind-lookup scheme to a
        :class:`_orm.Session` given a set of :class:`_engine.Engine` objects.
        To apply a corresponding :meth:`_orm.Session.get_bind` implementation
        for use with a :class:`.AsyncSession` and :class:`.AsyncEngine`
        objects, continue to subclass :class:`_orm.Session` and apply it to
        :class:`.AsyncSession` using
        :paramref:`.AsyncSession.sync_session_class`. The inner method must
        continue to return :class:`_engine.Engine` instances, which can be
        acquired from a :class:`_asyncio.AsyncEngine` using the
        :attr:`_asyncio.AsyncEngine.sync_engine` attribute::

            # using example from "Custom Vertical Partitioning"


            import random

            from sqlalchemy.ext.asyncio import AsyncSession
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy.ext.asyncio import async_sessionmaker
            from sqlalchemy.orm import Session

            # construct async engines w/ async drivers
            engines = {
                "leader": create_async_engine("sqlite+aiosqlite:///leader.db"),
                "other": create_async_engine("sqlite+aiosqlite:///other.db"),
                "follower1": create_async_engine("sqlite+aiosqlite:///follower1.db"),
                "follower2": create_async_engine("sqlite+aiosqlite:///follower2.db"),
            }


            class RoutingSession(Session):
                def get_bind(self, mapper=None, clause=None, **kw):
                    # within get_bind(), return sync engines
                    if mapper and issubclass(mapper.class_, MyOtherClass):
                        return engines["other"].sync_engine
                    elif self._flushing or isinstance(clause, (Update, Delete)):
                        return engines["leader"].sync_engine
                    else:
                        return engines[
                            random.choice(["follower1", "follower2"])
                        ].sync_engine


            # apply to AsyncSession using sync_session_class
            AsyncSessionMaker = async_sessionmaker(sync_session_class=RoutingSession)

        The :meth:`_orm.Session.get_bind` method is called in a non-asyncio,
        implicitly non-blocking context in the same manner as ORM event hooks
        and functions that are invoked via :meth:`.AsyncSession.run_sync`, so
        routines that wish to run SQL commands inside of
        :meth:`_orm.Session.get_bind` can continue to do so using
        blocking-style code, which will be translated to implicitly async calls
        at the point of invoking IO on the database drivers.


        """  # noqa: E501

        return self._proxied.get_bind(
            mapper=mapper, clause=clause, bind=bind, **kw
        )

    def is_modified(
        self, instance: object, include_collections: bool = True
    ) -> bool:
        r"""Return ``True`` if the given instance has locally
        modified attributes.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        This method retrieves the history for each instrumented
        attribute on the instance and performs a comparison of the current
        value to its previously flushed or committed value, if any.

        It is in effect a more expensive and accurate
        version of checking for the given instance in the
        :attr:`.Session.dirty` collection; a full test for
        each attribute's net "dirty" status is performed.

        E.g.::

            return session.is_modified(someobject)

        A few caveats to this method apply:

        * Instances present in the :attr:`.Session.dirty` collection may
          report ``False`` when tested with this method.  This is because
          the object may have received change events via attribute mutation,
          thus placing it in :attr:`.Session.dirty`, but ultimately the state
          is the same as that loaded from the database, resulting in no net
          change here.
        * Scalar attributes may not have recorded the previously set
          value when a new value was applied, if the attribute was not loaded,
          or was expired, at the time the new value was received - in these
          cases, the attribute is assumed to have a change, even if there is
          ultimately no net change against its database value. SQLAlchemy in
          most cases does not need the "old" value when a set event occurs, so
          it skips the expense of a SQL call if the old value isn't present,
          based on the assumption that an UPDATE of the scalar value is
          usually needed, and in those few cases where it isn't, is less
          expensive on average than issuing a defensive SELECT.

          The "old" value is fetched unconditionally upon set only if the
          attribute container has the ``active_history`` flag set to ``True``.
          This flag is set typically for primary key attributes and scalar
          object references that are not a simple many-to-one.  To set this
          flag for any arbitrary mapped column, use the ``active_history``
          argument with :func:`.column_property`.

        :param instance: mapped instance to be tested for pending changes.
        :param include_collections: Indicates if multivalued collections
         should be included in the operation.  Setting this to ``False`` is a
         way to detect only local-column based properties (i.e. scalar columns
         or many-to-one foreign keys) that would result in an UPDATE for this
         instance upon flush.



        """  # noqa: E501

        return self._proxied.is_modified(
            instance, include_collections=include_collections
        )

    async def invalidate(self) -> None:
        r"""Close this Session, using connection invalidation.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        For a complete description, see :meth:`_orm.Session.invalidate`.

        """  # noqa: E501

        return await self._proxied.invalidate()

    async def merge(
        self,
        instance: _O,
        *,
        load: bool = True,
        options: Optional[Sequence[ORMOption]] = None,
    ) -> _O:
        r"""Copy the state of a given instance into a corresponding instance
        within this :class:`_asyncio.AsyncSession`.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. seealso::

            :meth:`_orm.Session.merge` - main documentation for merge


        """  # noqa: E501

        return await self._proxied.merge(instance, load=load, options=options)

    async def refresh(
        self,
        instance: object,
        attribute_names: Optional[Iterable[str]] = None,
        with_for_update: ForUpdateParameter = None,
    ) -> None:
        r"""Expire and refresh the attributes on the given instance.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        A query will be issued to the database and all attributes will be
        refreshed with their current database value.

        This is the async version of the :meth:`_orm.Session.refresh` method.
        See that method for a complete description of all options.

        .. seealso::

            :meth:`_orm.Session.refresh` - main documentation for refresh


        """  # noqa: E501

        return await self._proxied.refresh(
            instance,
            attribute_names=attribute_names,
            with_for_update=with_for_update,
        )

    async def rollback(self) -> None:
        r"""Rollback the current transaction in progress.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. seealso::

            :meth:`_orm.Session.rollback` - main documentation for
            "rollback"

        """  # noqa: E501

        return await self._proxied.rollback()

    @overload
    async def scalar(
        self,
        statement: TypedReturnsRows[Tuple[_T]],
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> Optional[_T]: ...

    @overload
    async def scalar(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> Any: ...

    async def scalar(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> Any:
        r"""Execute a statement and return a scalar result.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. seealso::

            :meth:`_orm.Session.scalar` - main documentation for scalar


        """  # noqa: E501

        return await self._proxied.scalar(
            statement,
            params=params,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
            **kw,
        )

    @overload
    async def scalars(
        self,
        statement: TypedReturnsRows[Tuple[_T]],
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> ScalarResult[_T]: ...

    @overload
    async def scalars(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> ScalarResult[Any]: ...

    async def scalars(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> ScalarResult[Any]:
        r"""Execute a statement and return scalar results.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        :return: a :class:`_result.ScalarResult` object

        .. versionadded:: 1.4.24 Added :meth:`_asyncio.AsyncSession.scalars`

        .. versionadded:: 1.4.26 Added
           :meth:`_asyncio.async_scoped_session.scalars`

        .. seealso::

            :meth:`_orm.Session.scalars` - main documentation for scalars

            :meth:`_asyncio.AsyncSession.stream_scalars` - streaming version


        """  # noqa: E501

        return await self._proxied.scalars(
            statement,
            params=params,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
            **kw,
        )

    async def get(
        self,
        entity: _EntityBindKey[_O],
        ident: _PKIdentityArgument,
        *,
        options: Optional[Sequence[ORMOption]] = None,
        populate_existing: bool = False,
        with_for_update: ForUpdateParameter = None,
        identity_token: Optional[Any] = None,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
    ) -> Union[_O, None]:
        r"""Return an instance based on the given primary key identifier,
        or ``None`` if not found.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. seealso::

            :meth:`_orm.Session.get` - main documentation for get



        """  # noqa: E501

        result = await self._proxied.get(
            entity,
            ident,
            options=options,
            populate_existing=populate_existing,
            with_for_update=with_for_update,
            identity_token=identity_token,
            execution_options=execution_options,
        )
        return result

    async def get_one(
        self,
        entity: _EntityBindKey[_O],
        ident: _PKIdentityArgument,
        *,
        options: Optional[Sequence[ORMOption]] = None,
        populate_existing: bool = False,
        with_for_update: ForUpdateParameter = None,
        identity_token: Optional[Any] = None,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
    ) -> _O:
        r"""Return an instance based on the given primary key identifier,
        or raise an exception if not found.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        Raises :class:`_exc.NoResultFound` if the query selects no rows.

        ..versionadded: 2.0.22

        .. seealso::

            :meth:`_orm.Session.get_one` - main documentation for get_one


        """  # noqa: E501

        return await self._proxied.get_one(
            entity,
            ident,
            options=options,
            populate_existing=populate_existing,
            with_for_update=with_for_update,
            identity_token=identity_token,
            execution_options=execution_options,
        )

    @overload
    async def stream(
        self,
        statement: TypedReturnsRows[_T],
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> AsyncResult[_T]: ...

    @overload
    async def stream(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> AsyncResult[Any]: ...

    async def stream(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> AsyncResult[Any]:
        r"""Execute a statement and return a streaming
        :class:`_asyncio.AsyncResult` object.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.


        """  # noqa: E501

        return await self._proxied.stream(
            statement,
            params=params,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
            **kw,
        )

    @overload
    async def stream_scalars(
        self,
        statement: TypedReturnsRows[Tuple[_T]],
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> AsyncScalarResult[_T]: ...

    @overload
    async def stream_scalars(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> AsyncScalarResult[Any]: ...

    async def stream_scalars(
        self,
        statement: Executable,
        params: Optional[_CoreAnyExecuteParams] = None,
        *,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> AsyncScalarResult[Any]:
        r"""Execute a statement and return a stream of scalar results.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        :return: an :class:`_asyncio.AsyncScalarResult` object

        .. versionadded:: 1.4.24

        .. seealso::

            :meth:`_orm.Session.scalars` - main documentation for scalars

            :meth:`_asyncio.AsyncSession.scalars` - non streaming version


        """  # noqa: E501

        return await self._proxied.stream_scalars(
            statement,
            params=params,
            execution_options=execution_options,
            bind_arguments=bind_arguments,
            **kw,
        )

    @property
    def bind(self) -> Any:
        r"""Proxy for the :attr:`_asyncio.AsyncSession.bind` attribute
        on behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        """  # noqa: E501

        return self._proxied.bind

    @bind.setter
    def bind(self, attr: Any) -> None:
        self._proxied.bind = attr

    @property
    def dirty(self) -> Any:
        r"""The set of all persistent instances considered dirty.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class
            on behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_asyncio.AsyncSession` class.

        E.g.::

            some_mapped_object in session.dirty

        Instances are considered dirty when they were modified but not
        deleted.

        Note that this 'dirty' calculation is 'optimistic'; most
        attribute-setting or collection modification operations will
        mark an instance as 'dirty' and place it in this set, even if
        there is no net change to the attribute's value.  At flush
        time, the value of each attribute is compared to its
        previously saved value, and if there's no net change, no SQL
        operation will occur (this is a more expensive operation so
        it's only done at flush time).

        To check if an instance has actionable net changes to its
        attributes, use the :meth:`.Session.is_modified` method.



        """  # noqa: E501

        return self._proxied.dirty

    @property
    def deleted(self) -> Any:
        r"""The set of all instances marked as 'deleted' within this ``Session``

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class
            on behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_asyncio.AsyncSession` class.


        """  # noqa: E501

        return self._proxied.deleted

    @property
    def new(self) -> Any:
        r"""The set of all instances marked as 'new' within this ``Session``.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class
            on behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_asyncio.AsyncSession` class.


        """  # noqa: E501

        return self._proxied.new

    @property
    def identity_map(self) -> Any:
        r"""Proxy for the :attr:`_orm.Session.identity_map` attribute
        on behalf of the :class:`_asyncio.AsyncSession` class.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class
            on behalf of the :class:`_asyncio.scoping.async_scoped_session` class.


        """  # noqa: E501

        return self._proxied.identity_map

    @identity_map.setter
    def identity_map(self, attr: Any) -> None:
        self._proxied.identity_map = attr

    @property
    def is_active(self) -> Any:
        r"""True if this :class:`.Session` not in "partial rollback" state.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class
            on behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_asyncio.AsyncSession` class.

        .. versionchanged:: 1.4 The :class:`_orm.Session` no longer begins
           a new transaction immediately, so this attribute will be False
           when the :class:`_orm.Session` is first instantiated.

        "partial rollback" state typically indicates that the flush process
        of the :class:`_orm.Session` has failed, and that the
        :meth:`_orm.Session.rollback` method must be emitted in order to
        fully roll back the transaction.

        If this :class:`_orm.Session` is not in a transaction at all, the
        :class:`_orm.Session` will autobegin when it is first used, so in this
        case :attr:`_orm.Session.is_active` will return True.

        Otherwise, if this :class:`_orm.Session` is within a transaction,
        and that transaction has not been rolled back internally, the
        :attr:`_orm.Session.is_active` will also return True.

        .. seealso::

            :ref:`faq_session_rollback`

            :meth:`_orm.Session.in_transaction`



        """  # noqa: E501

        return self._proxied.is_active

    @property
    def autoflush(self) -> Any:
        r"""Proxy for the :attr:`_orm.Session.autoflush` attribute
        on behalf of the :class:`_asyncio.AsyncSession` class.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class
            on behalf of the :class:`_asyncio.scoping.async_scoped_session` class.


        """  # noqa: E501

        return self._proxied.autoflush

    @autoflush.setter
    def autoflush(self, attr: Any) -> None:
        self._proxied.autoflush = attr

    @property
    def no_autoflush(self) -> Any:
        r"""Return a context manager that disables autoflush.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class
            on behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_asyncio.AsyncSession` class.

        e.g.::

            with session.no_autoflush:

                some_object = SomeClass()
                session.add(some_object)
                # won't autoflush
                some_object.related_thing = session.query(SomeRelated).first()

        Operations that proceed within the ``with:`` block
        will not be subject to flushes occurring upon query
        access.  This is useful when initializing a series
        of objects which involve existing database queries,
        where the uncompleted object should not yet be flushed.



        """  # noqa: E501

        return self._proxied.no_autoflush

    @property
    def info(self) -> Any:
        r"""A user-modifiable dictionary.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class
            on behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class
            on behalf of the :class:`_asyncio.AsyncSession` class.

        The initial value of this dictionary can be populated using the
        ``info`` argument to the :class:`.Session` constructor or
        :class:`.sessionmaker` constructor or factory methods.  The dictionary
        here is always local to this :class:`.Session` and can be modified
        independently of all other :class:`.Session` objects.



        """  # noqa: E501

        return self._proxied.info

    @classmethod
    async def close_all(cls) -> None:
        r"""Close all :class:`_asyncio.AsyncSession` sessions.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. deprecated:: 2.0 The :meth:`.AsyncSession.close_all` method is deprecated and will be removed in a future release.  Please refer to :func:`_asyncio.close_all_sessions`.

        """  # noqa: E501

        return await AsyncSession.close_all()

    @classmethod
    def object_session(cls, instance: object) -> Optional[Session]:
        r"""Return the :class:`.Session` to which an object belongs.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        This is an alias of :func:`.object_session`.



        """  # noqa: E501

        return AsyncSession.object_session(instance)

    @classmethod
    def identity_key(
        cls,
        class_: Optional[Type[Any]] = None,
        ident: Union[Any, Tuple[Any, ...]] = None,
        *,
        instance: Optional[Any] = None,
        row: Optional[Union[Row[Any], RowMapping]] = None,
        identity_token: Optional[Any] = None,
    ) -> _IdentityKeyType[Any]:
        r"""Return an identity key.

        .. container:: class_bases

            Proxied for the :class:`_asyncio.AsyncSession` class on
            behalf of the :class:`_asyncio.scoping.async_scoped_session` class.

        .. container:: class_bases

            Proxied for the :class:`_orm.Session` class on
            behalf of the :class:`_asyncio.AsyncSession` class.

        This is an alias of :func:`.util.identity_key`.



        """  # noqa: E501

        return AsyncSession.identity_key(
            class_=class_,
            ident=ident,
            instance=instance,
            row=row,
            identity_token=identity_token,
        )

    # END PROXY METHODS async_scoped_session