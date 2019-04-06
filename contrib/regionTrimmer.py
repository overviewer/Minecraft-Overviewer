#!/usr/bin/env python3

"""Deletes outlying and unconnected regions"""

import argparse
import logging
from pathlib import Path

import networkx

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_region_file_from_node(regionset_path, node):
    return regionset_path / ('r.%d.%d.mca' % node)


def get_nodes(regionset_path):
    return [
        tuple(int(x) for x in r.stem.split('.')[1:3])
        for r in regionset_path.glob('r.*.*.mca')
    ]


def generate_edges(graph):
    offsets = (-1, 1)
    nodes = graph.nodes()
    for node in nodes:
        for offset in offsets:
            graph.add_edges_from(
                (node, offset_node)
                for offset_node in [
                    (node[0] + offset, node[1]),
                    (node[0], node[1] + offset),
                    (node[0] + offset, node[1] + offset),
                ]
                if offset_node in nodes
            )
    return graph


def generate_subgraphs(nodes):
    graph = networkx.Graph()
    graph.add_nodes_from(nodes)
    generate_edges(graph)
    return graph, [graph.subgraph(c) for c in networkx.connected_components(graph)]


def get_graph_bounds(graph):
    nodes = graph.nodes()
    return (
        max(n[0] for n in nodes),
        min(n[0] for n in nodes),
        max(n[1] for n in nodes),
        min(n[1] for n in nodes),
    )


def get_graph_center_by_bounds(bounds):
    dx = bounds[0] - bounds[1]
    dy = bounds[2] - bounds[3]
    return (dx // 2 + bounds[1], dy // 2 + bounds[3])


def trim_regions(graph, regions_path, dry_run=True, filter_func=lambda n: True):
    regions = [
        (n, get_region_file_from_node(regions_path, n))
        for n in graph.nodes()
        if filter_func(n)
    ]
    logger.info("Trimming regions: %s", ", ".join(x[1] for x in regions))
    for n, region_file in regions:
        graph.remove_node(n)
        if dry_run is False:
            unlink_file(region_file)


def is_outside_main(center, main_section_bounds):
    return center[0] <= main_section_bounds[0] and center[0] >= main_section_bounds[1] and \
        center[1] <= main_section_bounds[2] and center[1] >= main_section_bounds[3]


def is_outside_bounds(node, trim_center, trim_bounds):
    return node[0] >= trim_center[0] + trim_bounds[0] or \
        node[0] <= trim_center[0] - trim_bounds[0] or \
        node[1] >= trim_center[1] + trim_bounds[1] or \
        node[1] <= trim_center[1] - trim_bounds[1]

def unlink_file(path):
    try:
        path.unlink()
    except OSError as err:
        logger.warning("Unable to delete file: %s", path)
        logger.warning("Error recieved was: %s", err)


def main(args):
    for path in args.paths:
        logger.info("Using regionset path: %s", path)
        nodes = get_nodes(path)
        if not len(nodes):
            logger.error("Found no nodes, are you sure there are .mca files in %s ?",
                         path)
            return
        logger.info("Found %d nodes", len(nodes))
        logger.info("Generating graphing nodes...")
        graph, subgraphs = generate_subgraphs(nodes)
        assert len(graph.nodes()) == sum(len(sg.nodes()) for sg in subgraphs)
        if len(subgraphs) == 1:
            logger.warning("All regions are contiguous, the needful is done!")
            return
        logger.info("Found %d discrete region sections", len(subgraphs))
        subgraphs = sorted(subgraphs, key=lambda sg: len(sg), reverse=True)
        for i, sg in enumerate(subgraphs):
            logger.info("Region section #%02d: %04d nodes", i + 1, len(sg.nodes()))
            bounds = get_graph_bounds(sg)
            logger.info("Bounds: %d <-> %d x %d <-> %d", *get_graph_bounds(sg))
            center = get_graph_center_by_bounds(bounds)
            logger.info("Center: %d x %d", *center)

        main_section = subgraphs[0]
        main_section_bounds = get_graph_bounds(main_section)
        main_section_center = get_graph_center_by_bounds(main_section_bounds)
        logger.info("Using %d node graph as main section,", len(main_section.nodes()))
        satellite_sections = subgraphs[1:]
        for ss in satellite_sections:
            bounds = get_graph_bounds(ss)
            center = get_graph_center_by_bounds(bounds)
            logger.info(("Checking satellite section with %d nodes, "
                         "%d <-> %d x %d <-> %d bounds and %d x %d center"),
                        len(ss.nodes()), *(bounds + center))

            if args.trim_disconnected:
                trim_regions(ss, path, dry_run=args.dry_run)

            if args.trim_outside_main:
                if is_outside_main(ss, center, main_section_bounds):
                    logger.info("Section is outside main section bounds")
                    trim_regions(ss, path, dry_run=args.dry_run)
                else:
                    logger.info("Section falls inside main section bounds, ignoring")

            if args.trim_outside_bounds:
                logger.info("Checking regions outside specified bounds")
                trim_center = args.trim_outside_bounds.get("center", main_section_center)
                trim_bounds = args.trim_outside_bounds["bounds"]
                trim_regions(ss, path, dry_run=args.dry_run,
                             filter_func=lambda n: is_outside_bounds(n, trim_center, trim_bounds))


def dir_path(path):
    p = Path(path)
    if not p.is_dir():
        raise argparse.ArgumentTypeError("Not a valid directory path")
    return p


def center_bound(value):
    x = [int(v) for v in value.split(",")]
    if len(x) == 4:
        return {"center": x[:2], "bounds": x[2:]}
    elif len(x) == 2:
        return {"bounds": x}
    else:
        raise argparse.ArgumentTypeError("Invalid center/bound value")


if __name__ == "__main__":
    logging.basicConfig()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", metavar="<path/to/region/directory>", nargs="+", type=dir_path)
    parser.add_argument("-D", "--trim-disconnected", action="store_true",
                        default=False, help="Trim all disconnected regions")
    parser.add_argument("-M", "--trim-outside-main", action="store_true",
                        default=False, help="Trim disconnected regions outside main section bounds")
    parser.add_argument("-B", "--trim-outside-bounds",
                        metavar="[center_X,center_Y,]bound_X,bound_Y", type=center_bound,
                        help=("Trim outside given bounds "
                              "(given as [center_X,center_Y,]bound_X,bound_Y)"))
    parser.add_argument("-n", "--dry-run", action="store_true", default=False,
                        help="Don't actually delete anything")
    args = parser.parse_args()
    main(args)
