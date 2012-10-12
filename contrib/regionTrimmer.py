#!/usr/bin/env python

"""Deletes outlying and unconnected regions"""

import logging
import os
import sys
import glob

import networkx

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_region_file_from_node(regionset_path, node):
    return os.path.join(regionset_path, 'r.%d.%d.mca' % node)

def get_nodes(regionset_path):
    return [tuple(map(int, r.split('.')[1:3])) \
        for r in glob.glob(os.path.join(regionset_path, 'r.*.*.mca'))]

def generate_edges(graph):
    offsets = (-1, 1)
    nodes = graph.nodes()
    for node in nodes:
        for offset in offsets:
            graph.add_edges_from((node, offset_node) for offset_node in \
                [(node[0] + offset, node[1]), (node[0], node[1] + offset), \
                    (node[0] + offset, node[1] + offset)] \
                if offset_node in nodes)
    return graph

def generate_subgraphs(nodes):
    graph = networkx.Graph()
    graph.add_nodes_from(nodes)
    generate_edges(graph)
    return graph, networkx.connected_component_subgraphs(graph)

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
    return (dx / 2 + bounds[1], dy / 2 + bounds[3])

def main(*args, **options):
    if len(args) < 1:
        logger.error('Missing region directory argument')
        return
    for path in args:
        logger.info('Using regionset path: %s', path)
        nodes = get_nodes(path)
        if not len(nodes):
            logger.error('Found no nodes, are you sure there are .mca files in %s ?',
                path)
            return
        logger.info('Found %d nodes', len(nodes))
        logger.info('Generating graphing nodes...')
        graph, subgraphs = generate_subgraphs(nodes)
        assert len(graph.nodes()) == sum(len(sg.nodes()) for sg in subgraphs)
        if len(subgraphs) == 1:
            logger.warn('All regions are contiguous, the needful is done!')
            return
        logger.info('Found %d discrete region sections', len(subgraphs))
        subgraphs = sorted(subgraphs, key=lambda sg: len(sg), reverse=True)
        for i, sg in enumerate(subgraphs):
            logger.info('Region section #%02d: %04d nodes', i+1, len(sg.nodes()))
            bounds = get_graph_bounds(sg)
            logger.info('Bounds: %d <-> %d x %d <-> %d', *get_graph_bounds(sg))
            center = get_graph_center_by_bounds(bounds)
            logger.info('Center: %d x %d', *center)

        main_section = subgraphs[0]
        main_section_bounds = get_graph_bounds(main_section)
        main_section_center = get_graph_center_by_bounds(main_section_bounds)
        logger.info('Using %d node graph as main section,', len(main_section.nodes()))
        satellite_sections = subgraphs[1:]
        for ss in satellite_sections:
            bounds = get_graph_bounds(ss)
            center = get_graph_center_by_bounds(bounds)
            logger.info('Checking satellite section with %d nodes, %d <-> %d x %d <-> %d bounds and %d x %d center',
                len(ss.nodes()), *(bounds + center))
            if options['trim_disconnected']:
                logger.info('Trimming regions: %s', ', '.join(
                    get_region_file_from_node(path, n) for n in ss.nodes()))
                for n, region_file in ((n, get_region_file_from_node(path, n)) \
                    for n in ss.nodes()):
                        ss.remove_node(n)
                        if not options['dry_run']:
                            unlink_file(region_file)
            if options['trim_outside_main']:
                if center[0] <= main_section_bounds[0] and center[0] >= main_section_bounds[1] and \
                        center[1] <= main_section_bounds[2] and center[1] >= main_section_bounds[3]:
                    logger.info('Section falls inside main section bounds, ignoring')
                else:
                    logger.info('Section is outside main section bounds')
                    logger.info('Trimming regions: %s', ', '.join(
                        get_region_file_from_node(path, n) for n in ss.nodes()))
                    for n, region_file in ((n, get_region_file_from_node(path, n)) \
                        for n in ss.nodes()):
                            ss.remove_node(n)
                            if not options['dry_run']:
                                unlink_file(region_file)
            if options['trim_outside_bounds']:
                x = map(int, options['trim_outside_bounds'].split(','))
                if len(x) == 4:
                    trim_center = x[:2]
                    trim_bounds = x[2:]
                elif len(x) == 2:
                    trim_center = main_section_center
                    trim_bounds = x
                else:
                    logger.error('Invalid center/bound value: %s',
                        options['trim_outside_bounds'])
                    continue
                for node in ss.nodes():
                    if node[0] >= trim_center[0] + trim_bounds[0] or \
                            node[0] <= trim_center[0] - trim_bounds[0] or \
                            node[1] >= trim_center[1] + trim_bounds[1] or \
                            node[1] <= trim_center[1] - trim_bounds[1]:
                        region_file = get_region_file_from_node(path, node)
                        logger.info('Region falls outside specified bounds, trimming: %s',
                            region_file)
                        ss.remove_node(node)
                        if not options['dry_run']:
                            unlink_file(region_file)

def unlink_file(path):
    try:
        os.unlink(path)
    except OSError as err:
        logger.warn('Unable to delete file: %s', path)
        logger.warn('Error recieved was: %s', err)


if __name__ == '__main__':
    import optparse
    logging.basicConfig()
    parser = optparse.OptionParser(
        usage='Usage: %prog [options] <path/to/region/directory>')
    parser.add_option('-D', '--trim-disconnected', action='store_true', default=False,
        help='Trim all disconnected regions')
    parser.add_option('-M', '--trim-outside-main', action='store_true', default=False,
        help='Trim disconnected regions outside main section bounds')
    parser.add_option('-B', '--trim-outside-bounds', default=False,
        metavar='[center_X,center_Y,]bound_X,bound_Y',
        help='Trim outside given bounds (given as [center_X,center_Y,]bound_X,bound_Y)')
    parser.add_option('-n', '--dry-run', action='store_true', default=False,
        help='Don\'t actually delete anything')
    opts, args = parser.parse_args()
    main(*args, **vars(opts))
