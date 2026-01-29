import OverlayOp from 'jsts/org/locationtech/jts/operation/overlay/OverlayOp';
import BufferOp from 'jsts/org/locationtech/jts/operation/buffer/BufferOp';
import TopologyPreservingSimplifier from 'jsts/org/locationtech/jts/simplify/TopologyPreservingSimplifier';
import Polygonizer from 'jsts/org/locationtech/jts/operation/polygonize/Polygonizer';
import {
  to3dPoints,
  toPlanarPoint,
  toPlanarPoints,
} from '../../../domain/geometry/algorithms/util/plane';
import {
  isPointWithinRing,
  toJstsMultiPolygon,
  toMultiPolygon2,
  toJstsLineSegment,
  toPolygon2,
} from '../../../domain/geometry/algorithms/util/polygon';
import { getTriangles } from '../../../domain/geometry/extract-triangles';
import {
  LineSegment,
  LineSegment2,
  LinearRing,
  LinearRing2,
  Plane,
  Point,
  Point2,
  Polygon2,
  Triangle,
} from '../../../domain/geometry/geometric-types';
import { MagicShapeResult } from '../../../services/viewer/shape-drawing/one-click-shapes/magic-shape-types';
import { mergeShapes } from '../../../services/viewer/shape-drawing/one-click-shapes/polygon-union-shape';
import {
  SparkelCategory,
  getCategoriesFromSparkelCategories,
} from '../../../domain/category-mapping';
import {
  areSegmentsEqual,
  createMultipolygon,
  getSegmentPlaneIntersection,
  isNearlyParallelAndCloseToPlane,
} from './util';
import { getAllRelatedDbIdsForMagicPolygon } from 'src/domain/property-operations';

const ALLOWED_CATEGORIES = [
  'IfcWall',
  'IfcWallStandardCase',
  'IfcBeam',
  'IfcColumn',
  'IfcCovering',
  'IfcSlab',
  'IfcDoor',
  'IfcRoof',
  'IfcWindow',
  'IfcBuildingElementPart',
  'Revit Walls',
  'Revit Columns',
  'Revit Beams',
  'Revit Roofs',
  'Revit Windows',
  'Revit Doors',
  'Revit Slabs',
  'Revit Covers',
  'Revit Floors',
  'Revit Ceilings',
  'Parts',
];

/*
This function is used to generate rings based on the intersection of the shape with the model.

It works as follows:
  1. It first gets the bounding box of the one-click-shape.
  2. It then finds all the potentially intersecting elements in the model based on the bounding box of the initial one-click-shape.
  3. It then calculates the intersection segments between the shape and the potentially intersecting meshes.
     These segments are then turned into rings and merged into a polygon.
  4. It then calculates the difference between the initial shape and the intersection polygon.
  5. Finally, it returns the difference as a set of rings that contains the hit point of the initial shape.
*/

const bufferAmount = 1e-2;
const BOUNDING_BOX_BUFFER = 1; // Buffer to catch walls with small gaps
// For parallel plane check - walls that are close to the plane should be included
// Based on logs, walls can be 3-5 units away, so using 10 units as tolerance
// This will catch walls on parallel planes with small offsets
// Toggle to log detailed timing information for Magic Polygon intersections
const DEBUG_MAGIC_SHAPE_TIMING = false;
// Verbose console logging for segments / rings per mesh
const DEBUG_MAGIC_SHAPE_LOG_DETAILS = false;
// Visualize raw intersection segments before ring construction
const DEBUG_MAGIC_SHAPE_VISUALIZE_SEGMENTS = false;
// Visualize rings after per-mesh merge
const DEBUG_MAGIC_SHAPE_VISUALIZE_GREEDY_RINGS = false;
// Visualize final rings after boolean difference & hit-point selection
const DEBUG_MAGIC_SHAPE_VISUALIZE_RINGS = false;

type IntersectionCacheEntry = {
  hasIntersections: boolean;
  intersectionMultipolygon?: Polygon2[];
};

const intersectionCache = new Map<string, IntersectionCacheEntry>();

export const invalidateMagicShapeIntersectionCache = (): void => {
  intersectionCache.clear();
};

const quantize = (value: number, decimals: number): number => {
  const factor = Math.pow(10, decimals);
  return Math.round(value * factor) / factor;
};

const buildIntersectionCacheKey = ({
  boundingBox,
  plane,
  hiddenModelUrns,
  allowedCategories,
  modelUrns,
  relatedDbIds,
}: {
  boundingBox: THREE.Box3;
  plane: Plane;
  hiddenModelUrns: string[];
  allowedCategories?: string[];
  modelUrns: string[];
  relatedDbIds: number[];
}): string => {
  const min = boundingBox.min;
  const max = boundingBox.max;

  const keyObject = {
    boundingBox: {
      min: [quantize(min.x, 2), quantize(min.y, 2), quantize(min.z, 2)],
      max: [quantize(max.x, 2), quantize(max.y, 2), quantize(max.z, 2)],
    },
    plane: {
      normal: plane.unitNormal.map((c) => quantize(c, 3)),
      coefficient: quantize(plane.planeCoefficient, 2),
    },
    hiddenModelUrns: [...hiddenModelUrns].sort(),
    allowedCategories: allowedCategories ? [...allowedCategories].sort() : null,
    modelUrns: [...modelUrns].sort(),
    relatedDbIds: [...relatedDbIds].sort((a, b) => a - b),
  };

  return JSON.stringify(keyObject);
};

export async function getRingsFromIntersections(
  shape: NonNullable<MagicShapeResult>,
  viewer: Autodesk.Viewing.Viewer3D,
  hiddenModelUrns: string[],
  selectedSparkelCategories?: string[]
): Promise<LinearRing[] | undefined> {
  const debugTimings: { label: string; elapsedMs: number }[] = [];
  const timeSection = <T>(label: string, fn: () => T): T => {
    if (!DEBUG_MAGIC_SHAPE_TIMING) {
      return fn();
    }
    const start = performance.now();
    const result = fn();
    const end = performance.now();
    debugTimings.push({ label, elapsedMs: end - start });
    return result;
  };

  const timeSectionAsync = async <T>(
    label: string,
    fn: () => Promise<T>
  ): Promise<T> => {
    if (!DEBUG_MAGIC_SHAPE_TIMING) {
      return fn();
    }
    const start = performance.now();
    const result = await fn();
    const end = performance.now();
    debugTimings.push({ label, elapsedMs: end - start });
    return result;
  };

  return executeRingsFromIntersections(
    shape,
    viewer,
    hiddenModelUrns,
    selectedSparkelCategories,
    debugTimings,
    timeSection,
    timeSectionAsync
  );
}

async function executeRingsFromIntersections(
  shape: NonNullable<MagicShapeResult>,
  viewer: Autodesk.Viewing.Viewer3D,
  hiddenModelUrns: string[],
  selectedSparkelCategories: string[] | undefined,
  debugTimings: { label: string; elapsedMs: number }[],
  timeSection: <T>(label: string, fn: () => T) => T,
  timeSectionAsync: <T>(label: string, fn: () => Promise<T>) => Promise<T>
): Promise<LinearRing[] | undefined> {
  try {
    const { relatedDbIds, allowedCategories, expandedBoundingBox, modelUrns } =
      await prepareIntersectionInputs(
        shape,
        viewer,
        hiddenModelUrns,
        selectedSparkelCategories,
        timeSection
      );

    const cacheEntry = await getOrComputeIntersectionCacheEntry(
      shape,
      viewer,
      relatedDbIds,
      allowedCategories,
      expandedBoundingBox,
      modelUrns,
      hiddenModelUrns,
      timeSection,
      timeSectionAsync
    );

    return processIntersectionResult(
      shape,
      viewer,
      cacheEntry,
      debugTimings,
      timeSection
    );
  } catch (error) {
    if (DEBUG_MAGIC_SHAPE_TIMING && debugTimings.length > 0) {
      // eslint-disable-next-line no-console
      console.log('MagicPolygon timings (error case):', debugTimings);
    }
    // eslint-disable-next-line no-console
    console.error('MagicPolygon getRingsFromIntersections failed', error);
    return shape.rings;
  }
}

function processIntersectionResult(
  shape: NonNullable<MagicShapeResult>,
  viewer: Autodesk.Viewing.Viewer3D,
  cacheEntry: IntersectionCacheEntry,
  debugTimings: { label: string; elapsedMs: number }[],
  timeSection: <T>(label: string, fn: () => T) => T
): LinearRing[] | undefined {
  if (!cacheEntry.hasIntersections) {
    return shape.rings;
  }

  const differenceAsMultiPolygon2 = computeDifferenceMultiPolygon(
    shape,
    cacheEntry,
    timeSection
  );

  const polygonContainingPoint = findPolygonContainingPoint(
    shape,
    differenceAsMultiPolygon2.polygons
  );

  const ringsFromIntersections = polygonContainingPoint
    ? [polygonContainingPoint.exterior, ...polygonContainingPoint.interiors]
    : undefined;

  if (
    DEBUG_MAGIC_SHAPE_VISUALIZE_RINGS &&
    ringsFromIntersections &&
    ringsFromIntersections.length > 0
  ) {
    visualizeIntersectionRings(
      ringsFromIntersections,
      shape.plane,
      viewer,
      'magic-shape-intersections-debug'
    );
  }

  if (DEBUG_MAGIC_SHAPE_TIMING && debugTimings.length > 0) {
    // Log timings grouped under a single label to avoid noisy logs
    // eslint-disable-next-line no-console
    console.log('MagicPolygon timings:', debugTimings);
  }

  if (ringsFromIntersections && ringsFromIntersections.length > 0) {
    return ringsFromIntersections.map(
      (ring) => to3dPoints(ring, shape.plane) as LinearRing
    );
  }

  return undefined;
}

async function prepareIntersectionInputs(
  shape: NonNullable<MagicShapeResult>,
  viewer: Autodesk.Viewing.Viewer3D,
  hiddenModelUrns: string[],
  selectedSparkelCategories: string[] | undefined,
  timeSection: <T>(label: string, fn: () => T) => T
): Promise<{
  relatedDbIds: number[];
  allowedCategories: string[] | undefined;
  expandedBoundingBox: THREE.Box3;
  modelUrns: string[];
}> {
  const rawRelatedDbIds = await getAllRelatedDbIdsForMagicPolygon(
    shape.hitTest.model,
    shape.hitTest.dbId
  );

  // Always ensure the originally clicked element itself is also ignored in the
  // intersection search, even if it wasn't returned from getAllRelatedDbIds
  // (for example due to hierarchy differences between IFC and Revit models).
  const relatedDbIdSet = new Set<number>(rawRelatedDbIds);
  relatedDbIdSet.add(shape.hitTest.dbId);
  const relatedDbIds = Array.from(relatedDbIdSet);

  if (DEBUG_MAGIC_SHAPE_LOG_DETAILS) {
    // eslint-disable-next-line no-console
    console.log('MagicPolygon: ignoring related dbIds', relatedDbIds);
  }

  const allowedCategories =
    selectedSparkelCategories && selectedSparkelCategories.length > 0
      ? getCategoriesFromSparkelCategories(
          selectedSparkelCategories as SparkelCategory[]
        )
      : undefined;

  const baseBoundingBox = getBoundingBox(shape.rings.flatMap((ring) => ring));

  const expandedBoundingBox = timeSection(
    'computeExpandedBoundingBox',
    () =>
      new THREE.Box3(
        baseBoundingBox.min.clone().subScalar(BOUNDING_BOX_BUFFER),
        baseBoundingBox.max.clone().addScalar(BOUNDING_BOX_BUFFER)
      )
  );

  const modelUrns = viewer.getAllModels().map((model) => model.getData().urn);

  return { relatedDbIds, allowedCategories, expandedBoundingBox, modelUrns };
}

async function getOrComputeIntersectionCacheEntry(
  shape: NonNullable<MagicShapeResult>,
  viewer: Autodesk.Viewing.Viewer3D,
  relatedDbIds: number[],
  allowedCategories: string[] | undefined,
  expandedBoundingBox: THREE.Box3,
  modelUrns: string[],
  hiddenModelUrns: string[],
  timeSection: <T>(label: string, fn: () => T) => T,
  timeSectionAsync: <T>(label: string, fn: () => Promise<T>) => Promise<T>
): Promise<IntersectionCacheEntry> {
  const cacheKey = buildIntersectionCacheKey({
    boundingBox: expandedBoundingBox,
    plane: shape.plane,
    hiddenModelUrns,
    allowedCategories,
    modelUrns,
    relatedDbIds,
  });

  let cacheEntry = intersectionCache.get(cacheKey);

  if (!cacheEntry) {
    const potentiallyIntersectingMeshes = await timeSectionAsync(
      'findIntersectingBoxesWithAllowedCategories',
      () =>
        findIntersectingBoxesWithAllowedCategories(
          viewer,
          expandedBoundingBox,
          shape.plane,
          relatedDbIds,
          hiddenModelUrns,
          allowedCategories
        )
    );

    if (potentiallyIntersectingMeshes.length === 0) {
      cacheEntry = { hasIntersections: false };
      intersectionCache.set(cacheKey, cacheEntry);
      return cacheEntry;
    }

    const intersectionContours = timeSection('getIntersectionContours', () =>
      getIntersectionContours(shape, potentiallyIntersectingMeshes, viewer)
    );
    const mergedIntersectionContours = timeSection(
      'mergeIntersectionContours',
      () => mergeShapes(intersectionContours)
    );

    const intersectionMultipolygon = createMultipolygon(
      mergedIntersectionContours
    );

    if (DEBUG_MAGIC_SHAPE_LOG_DETAILS) {
      // eslint-disable-next-line no-console
      console.log('MagicPolygon: global merged intersection multipolygon', {
        polygonCount: intersectionMultipolygon.polygons.length,
      });
    }

    cacheEntry = {
      hasIntersections: true,
      intersectionMultipolygon: intersectionMultipolygon.polygons,
    };
    intersectionCache.set(cacheKey, cacheEntry);
  }

  return cacheEntry;
}

function computeDifferenceMultiPolygon(
  shape: NonNullable<MagicShapeResult>,
  cacheEntry: IntersectionCacheEntry,
  timeSection: <T>(label: string, fn: () => T) => T
) {
  return timeSection('booleanOpsAndSimplify', () => {
    const planarShape = shape.rings.map((ring) =>
      toPlanarPoints(ring, shape.plane)
    ) as LinearRing2[];

    const shapeAsPolygon = createMultipolygon(planarShape);
    let shapeAsJSTSPolygon = toJstsMultiPolygon(shapeAsPolygon);

    const intersectionMultipolygon2: { polygons: Polygon2[] } = {
      polygons: cacheEntry.intersectionMultipolygon ?? [],
    };

    const intersectionPolygon = toJstsMultiPolygon(intersectionMultipolygon2);

    if (DEBUG_MAGIC_SHAPE_LOG_DETAILS) {
      // eslint-disable-next-line no-console
      console.log('MagicPolygon: boolean diff inputs', {
        shapeArea: shapeAsJSTSPolygon.getArea(),
        intersectionArea: intersectionPolygon.getArea(),
      });
    }

    const bufferedIntersectionPolygon = BufferOp.bufferOp(
      intersectionPolygon,
      bufferAmount,
      0
    );

    if (DEBUG_MAGIC_SHAPE_LOG_DETAILS) {
      // eslint-disable-next-line no-console
      console.log('MagicPolygon: buffered intersection', {
        area: bufferedIntersectionPolygon.getArea(),
      });
    }

    shapeAsJSTSPolygon = OverlayOp.difference(
      shapeAsJSTSPolygon,
      bufferedIntersectionPolygon
    );

    if (DEBUG_MAGIC_SHAPE_LOG_DETAILS) {
      // eslint-disable-next-line no-console
      console.log('MagicPolygon: diff result', {
        area: shapeAsJSTSPolygon.getArea(),
      });
    }

    const simplifier = new TopologyPreservingSimplifier(shapeAsJSTSPolygon);
    simplifier.setDistanceTolerance(bufferAmount * 2);
    const simplified = simplifier.getResultGeometry();

    return toMultiPolygon2(simplified);
  });
}

const getBoundingBox = (points: Point[], tolerance = 0.1): THREE.Box3 => {
  const minX = Math.min(...points.map((point) => point[0]));
  const maxX = Math.max(...points.map((point) => point[0]));
  const minY = Math.min(...points.map((point) => point[1]));
  const maxY = Math.max(...points.map((point) => point[1]));
  const minZ = Math.min(...points.map((point) => point[2]));
  const maxZ = Math.max(...points.map((point) => point[2]));

  return new THREE.Box3(
    new THREE.Vector3(minX - tolerance, minY - tolerance, minZ - tolerance),
    new THREE.Vector3(maxX + tolerance, maxY + tolerance, maxZ + tolerance)
  );
};

function getIntersectionContours(
  shape: NonNullable<MagicShapeResult>,
  potentiallyIntersectingMeshes: { dbId: number; triangles: Triangle[] }[],
  viewer: Autodesk.Viewing.Viewer3D
): LinearRing2[] {
  const intersectionContours: LinearRing2[] = [];

  for (const mesh of potentiallyIntersectingMeshes) {
    const mergedRingsForMesh = getIntersectionContoursForMesh(
      shape,
      mesh,
      viewer
    );
    intersectionContours.push(...mergedRingsForMesh);
  }

  return intersectionContours;
}

function getIntersectionContoursForMesh(
  shape: NonNullable<MagicShapeResult>,
  mesh: { dbId: number; triangles: Triangle[] },
  viewer: Autodesk.Viewing.Viewer3D
): LinearRing2[] {
  const intersectionSegments = getIntersectionSegments(
    mesh.triangles,
    shape.plane
  );

  if (intersectionSegments.length < 3) {
    return [];
  }

  if (DEBUG_MAGIC_SHAPE_LOG_DETAILS) {
    // eslint-disable-next-line no-console
    console.log('MagicPolygon: mesh segments', {
      dbId: mesh.dbId,
      segmentCount: intersectionSegments.length,
    });
  }

  if (DEBUG_MAGIC_SHAPE_VISUALIZE_SEGMENTS && intersectionSegments.length > 0) {
    visualizeIntersectionSegments(
      intersectionSegments,
      shape.plane,
      viewer,
      'magic-shape-intersections-segments'
    );
  }

  const deduplicatedSegments = removeDuplicateSegments(intersectionSegments);
  if (DEBUG_MAGIC_SHAPE_LOG_DETAILS) {
    // eslint-disable-next-line no-console
    console.log('MagicPolygon: deduplicated segments', {
      dbId: mesh.dbId,
      segmentCount: deduplicatedSegments.length,
    });
  }

  const rings = createRingsFromSegments(deduplicatedSegments);

  if (DEBUG_MAGIC_SHAPE_LOG_DETAILS) {
    // eslint-disable-next-line no-console
    console.log('MagicPolygon: rings before merge', {
      dbId: mesh.dbId,
      ringCount: rings.length,
      ringLengths: rings.map((r) => r.length),
    });
  }

  const mergedRings = mergeShapes(rings);

  if (DEBUG_MAGIC_SHAPE_LOG_DETAILS) {
    // eslint-disable-next-line no-console
    console.log('MagicPolygon: merged rings (per mesh)', {
      dbId: mesh.dbId,
      ringCount: mergedRings.length,
      ringLengths: mergedRings.map((r) => r.length),
    });
  }

  if (DEBUG_MAGIC_SHAPE_VISUALIZE_GREEDY_RINGS && mergedRings.length > 0) {
    visualizeIntersectionRings(
      mergedRings,
      shape.plane,
      viewer,
      'magic-shape-intersections-greedy'
    );
  }

  return mergedRings;
}

function removeDuplicateSegments(
  segments: LineSegment2[],
  tolerance = 0.01
): LineSegment2[] {
  let uniqueSegments: LineSegment2[] = []; // Will store one copy of each unique segment

  for (const segment of segments) {
    // Flag to check if current segment is a duplicate
    let isDuplicate = false;

    for (let i = 0; i < uniqueSegments.length; i++) {
      if (areSegmentsEqual(segment, uniqueSegments[i], tolerance)) {
        isDuplicate = true;
        break;
      }
    }

    // If not a duplicate, add it to the list of unique segments
    if (!isDuplicate) {
      uniqueSegments.push(segment);
    }
  }

  return uniqueSegments;
}

function createRingsFromSegments(contours: LineSegment2[]): LinearRing2[] {
  const polygonizer = new Polygonizer();

  // Snap coordinates to a fixed grid before feeding to Polygonizer so that
  // endpoints that are conceptually the same become bit-identical.
  const snapPrecision = 5;
  const snapPoint = (p: Point2): Point2 => [
    Number(p[0].toFixed(snapPrecision)),
    Number(p[1].toFixed(snapPrecision)),
  ];

  const snappedContours: LineSegment2[] = contours
    .map<LineSegment2>(([a, b]) => [snapPoint(a), snapPoint(b)])
    // Drop segments that collapse to a point after snapping
    .filter(
      ([a, b]) => Math.abs(a[0] - b[0]) > 1e-8 || Math.abs(a[1] - b[1]) > 1e-8
    );

  snappedContours.forEach((segment) => {
    const line = toJstsLineSegment(segment);
    polygonizer.add(line);
  });

  const polys = polygonizer.getPolygons();
  const rings: LinearRing2[] = [];

  const polysArray =
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (polys as any).toArray?.() ??
    // Fallback if toArray is not available
    [];

  polysArray.forEach((poly: unknown) => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const polygon2 = toPolygon2(poly as any);
    rings.push(polygon2.exterior, ...polygon2.interiors);
  });

  if (DEBUG_MAGIC_SHAPE_LOG_DETAILS) {
    // eslint-disable-next-line no-console
    console.log('MagicPolygon: createRingsFromSegments polygonizer result', {
      ringCount: rings.length,
      ringLengths: rings.map((r) => r.length),
    });
  }

  return rings;
}

function visualizeIntersectionRings(
  rings: LinearRing2[],
  plane: Plane,
  viewer: Autodesk.Viewing.Viewer3D,
  overlayName: string
): void {
  const materials: THREE.MeshBasicMaterial[] = [];
  const meshes: THREE.Mesh[] = [];

  viewer.overlays.addScene(overlayName);

  const baseColors = [
    0xe53e3e, // red
    0xdd6b20, // orange
    0xd69e2e, // yellow
    0x38a169, // green
    0x3182ce, // blue
    0x805ad5, // purple
    0xd53f8c, // pink
  ];

  const to3d = (ring: LinearRing2): Point[] =>
    to3dPoints(ring, plane) as Point[];

  rings.forEach((ring, ringIndex) => {
    const colorHex = baseColors[ringIndex % baseColors.length];
    const material = new THREE.MeshBasicMaterial({
      color: new THREE.Color().setHex(colorHex),
      opacity: 0.9,
      transparent: true,
      depthTest: false,
    });
    materials.push(material);

    const ring3d = to3d(ring);

    for (let i = 0; i < ring3d.length - 1; i++) {
      const start = ring3d[i];
      const end = ring3d[i + 1];

      const startVec = new THREE.Vector3(start[0], start[1], start[2]);
      const endVec = new THREE.Vector3(end[0], end[1], end[2]);
      const segmentVec = new THREE.Vector3().subVectors(endVec, startVec);
      const length = segmentVec.length();

      if (length === 0) {
        continue;
      }

      const radius = 0.01;
      const geometry = new THREE.CylinderGeometry(radius, radius, length, 8);

      const mesh = new THREE.Mesh(geometry, material);

      // Orient cylinder along segment
      const axis = new THREE.Vector3(0, 1, 0);
      const dir = segmentVec.clone().normalize();
      const quaternion = new THREE.Quaternion().setFromUnitVectors(axis, dir);
      mesh.quaternion.copy(quaternion);

      // Position at midpoint
      const midpoint = new THREE.Vector3()
        .addVectors(startVec, endVec)
        .multiplyScalar(0.5);
      mesh.position.copy(midpoint);

      viewer.overlays.addMesh(mesh, overlayName);
      meshes.push(mesh);
    }

    const handleOnce = () => {
      meshes.forEach((mesh) => {
        viewer.overlays.removeMesh(mesh, overlayName);
        mesh.geometry.dispose();
      });
      materials.forEach((materialInstance) => materialInstance.dispose());
      viewer.removeEventListener(
        Autodesk.Viewing.AGGREGATE_ISOLATION_CHANGED_EVENT,
        handleOnce
      );
    };

    viewer.addEventListener(
      Autodesk.Viewing.AGGREGATE_ISOLATION_CHANGED_EVENT,
      handleOnce
    );
  });

  // Note: for rings we currently leave meshes in place; they are transient
  // debug overlays that get cleared when a new magic polygon is drawn.
}

function visualizeIntersectionSegments(
  segments: LineSegment2[],
  plane: Plane,
  viewer: Autodesk.Viewing.Viewer3D,
  overlayName: string
): void {
  const materials: THREE.MeshBasicMaterial[] = [];
  const meshes: THREE.Mesh[] = [];

  viewer.overlays.addScene(overlayName);

  const material = new THREE.MeshBasicMaterial({
    color: new THREE.Color().setHex(0x0000ff),
    opacity: 0.9,
    transparent: true,
    depthTest: false,
  });
  materials.push(material);

  const to3d = (segment: LineSegment2): [Point, Point] =>
    to3dPoints(segment, plane) as [Point, Point];

  segments.forEach((segment) => {
    const [start, end] = to3d(segment);

    const startVec = new THREE.Vector3(start[0], start[1], start[2]);
    const endVec = new THREE.Vector3(end[0], end[1], end[2]);
    const segmentVec = new THREE.Vector3().subVectors(endVec, startVec);
    const length = segmentVec.length();

    if (length === 0) {
      return;
    }

    const radius = 0.01;
    const geometry = new THREE.CylinderGeometry(radius, radius, length, 8);

    const mesh = new THREE.Mesh(geometry, material);

    const axis = new THREE.Vector3(0, 1, 0);
    const dir = segmentVec.clone().normalize();
    const quaternion = new THREE.Quaternion().setFromUnitVectors(axis, dir);
    mesh.quaternion.copy(quaternion);

    const midpoint = new THREE.Vector3()
      .addVectors(startVec, endVec)
      .multiplyScalar(0.5);
    mesh.position.copy(midpoint);

    viewer.overlays.addMesh(mesh, overlayName);
    meshes.push(mesh);
  });

  const handleOnce = () => {
    meshes.forEach((mesh) => {
      viewer.overlays.removeMesh(mesh, overlayName);
      mesh.geometry.dispose();
    });
    materials.forEach((materialInstance) => materialInstance.dispose());
    viewer.removeEventListener(
      Autodesk.Viewing.AGGREGATE_ISOLATION_CHANGED_EVENT,
      handleOnce
    );
  };

  viewer.addEventListener(
    Autodesk.Viewing.AGGREGATE_ISOLATION_CHANGED_EVENT,
    handleOnce
  );
}

// eslint-disable-next-line complexity
async function findIntersectingBoxesWithAllowedCategories(
  viewer: Autodesk.Viewing.Viewer3D,
  boundingBox: THREE.Box3,
  plane: Plane,
  dbIdsToIgnore: number[],
  hiddenModelUrns: string[],
  allowedCategories?: string[]
): Promise<{ dbId: number; triangles: Triangle[] }[]> {
  // Use provided categories or fall back to default
  const categoriesToUse =
    allowedCategories && allowedCategories.length > 0
      ? allowedCategories
      : ALLOWED_CATEGORIES;
  const allowedCategoriesSet = new Set(categoriesToUse);
  const potentiallyIntersectingMeshes: {
    dbId: number;
    triangles: Triangle[];
  }[] = [];

  for (const model of viewer.getAllModels()) {
    if (hiddenModelUrns.includes(model.getData().urn)) {
      console.log(`Skipping hidden model ${model.getData().name}`);
      continue;
    }
    const boundingBoxesCoordinates = model.getData().fragments
      .boxes as number[];
    const fragId2dbId = model.getData().fragments.fragId2dbId as number[];

    const instanceTree = model.getData().instanceTree;
    const candidateDbIds = new Set<number>();

    // First pass: identify candidate dbIds whose fragment bounding boxes intersect the query box
    // and that are visible / not ignored. This keeps subsequent property & triangle work bounded.
    for (let i = 0; i < boundingBoxesCoordinates.length; i += 6) {
      const fragId = i / 6;
      const dbId = fragId2dbId[fragId];

      if (dbIdsToIgnore.includes(dbId) || !viewer.isNodeVisible(dbId, model)) {
        continue;
      }

      const boxMin = new THREE.Vector3(
        boundingBoxesCoordinates[i],
        boundingBoxesCoordinates[i + 1],
        boundingBoxesCoordinates[i + 2]
      );
      const boxMax = new THREE.Vector3(
        boundingBoxesCoordinates[i + 3],
        boundingBoxesCoordinates[i + 4],
        boundingBoxesCoordinates[i + 5]
      );

      const fragBox = new THREE.Box3(boxMin, boxMax);

      if (fragBox.intersectsBox(boundingBox)) {
        candidateDbIds.add(dbId);
      }
    }

    if (candidateDbIds.size === 0) {
      continue;
    }

    // Collect candidate dbIds plus their ancestors so we can mimic the parent lookup behaviour
    // from getPropertyByDbId without issuing separate property requests per node.
    const dbIdsForProperties = new Set<number>(candidateDbIds);

    if (instanceTree) {
      for (const dbId of candidateDbIds) {
        let currentId: number | null = dbId;
        const visited = new Set<number>();
        // Limit ancestry walk to avoid any pathological hierarchies
        const maxDepth = 32;
        let depth = 0;

        while (
          currentId !== null &&
          !visited.has(currentId) &&
          depth < maxDepth
        ) {
          visited.add(currentId);
          dbIdsForProperties.add(currentId);
          const parentId: number | null =
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (instanceTree as any).getNodeParentId &&
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (instanceTree as any).getNodeParentId(currentId) !== undefined
              ? // eslint-disable-next-line @typescript-eslint/no-explicit-any
                (instanceTree as any).getNodeParentId(currentId)
              : null;
          if (parentId === null || parentId === currentId) {
            break;
          }
          currentId = parentId;
          depth += 1;
        }
      }
    }

    // Bulk-fetch the relevant category properties once per model, instead of per dbId.
    const propertyResults: Autodesk.Viewing.PropertyResult[] =
      await new Promise((resolve, reject) => {
        model.getBulkProperties2(
          Array.from(dbIdsForProperties),
          {
            propFilter: ['IfcClass', 'Category'],
            categoryFilter: undefined,
          },
          (res) => resolve(res),
          () =>
            reject(new Error('Failed to get bulk properties for MagicPolygon'))
        );
      });

    const ifcClassByDbId = new Map<number, string>();
    const revitCategoryByDbId = new Map<number, string>();

    for (const props of propertyResults) {
      for (const prop of props.properties) {
        if (prop.attributeName === 'IfcClass' && prop.displayValue != null) {
          ifcClassByDbId.set(props.dbId, prop.displayValue.toString());
        }
        if (
          prop.attributeName === 'Category' &&
          prop.displayValue != null &&
          prop.displayCategory === '__category__'
        ) {
          revitCategoryByDbId.set(props.dbId, prop.displayValue.toString());
        }
      }
    }

    const getParentIdSafe = (
      tree: Autodesk.Viewing.InstanceTree,
      nodeId: number
    ): number | null => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const anyTree = tree as any;
      if (!anyTree.getNodeParentId) {
        return null;
      }
      const parent = anyTree.getNodeParentId(nodeId);
      return typeof parent === 'number' ? parent : null;
    };

    const getIfcClassAndCategoryForDbId = (
      dbId: number
    ): {
      ifcClass?: string;
      revitCategory?: string;
    } => {
      let currentId: number | null = dbId;
      const visited = new Set<number>();
      const maxDepth = 32;
      let depth = 0;

      while (
        currentId !== null &&
        !visited.has(currentId) &&
        depth < maxDepth
      ) {
        visited.add(currentId);
        const ifcClass = ifcClassByDbId.get(currentId);
        const revitCategory = revitCategoryByDbId.get(currentId);

        if (ifcClass || revitCategory) {
          return { ifcClass, revitCategory };
        }

        if (!instanceTree) {
          break;
        }

        const parentId = getParentIdSafe(instanceTree, currentId);
        if (parentId === null || parentId === currentId) {
          break;
        }
        currentId = parentId;
        depth += 1;
      }

      return {};
    };

    for (const dbId of candidateDbIds) {
      const { ifcClass, revitCategory } = getIfcClassAndCategoryForDbId(dbId);

      const isAllowedIfcClass =
        !!ifcClass && allowedCategoriesSet.has(ifcClass);
      const isAllowedRevitCategory =
        !!revitCategory && allowedCategoriesSet.has(revitCategory);

      if (DEBUG_MAGIC_SHAPE_LOG_DETAILS) {
        // eslint-disable-next-line no-console
        console.log('MagicPolygon: candidate categories', {
          modelName: model.getData().name,
          dbId,
          ifcClass,
          revitCategory,
          isAllowedIfcClass,
          isAllowedRevitCategory,
        });
      }

      if (isAllowedIfcClass || isAllowedRevitCategory) {
        const triangles = getTriangles(model, dbId);
        potentiallyIntersectingMeshes.push({ dbId, triangles });
      }
    }
  }

  return potentiallyIntersectingMeshes;
}

function getIntersectionSegments(
  elementTriangles: Triangle[],
  plane: Plane
): LineSegment2[] {
  const intersectionLines: LineSegment2[] = [];

  const planeNormal = plane.unitNormal;
  const planeCoefficient = plane.planeCoefficient;
  const planeTolerance = 0.05; // Distance tolerance for deciding if a triangle is close enough to intersect

  const triangleIntersectsPlane = (triangle: Triangle): boolean => {
    const [p0, p1, p2] = triangle;
    const d0 =
      p0[0] * planeNormal[0] +
      p0[1] * planeNormal[1] +
      p0[2] * planeNormal[2] +
      planeCoefficient;
    const d1 =
      p1[0] * planeNormal[0] +
      p1[1] * planeNormal[1] +
      p1[2] * planeNormal[2] +
      planeCoefficient;
    const d2 =
      p2[0] * planeNormal[0] +
      p2[1] * planeNormal[1] +
      p2[2] * planeNormal[2] +
      planeCoefficient;

    const sameSign =
      (d0 >= 0 && d1 >= 0 && d2 >= 0) || (d0 <= 0 && d1 <= 0 && d2 <= 0);
    const minAbsDistance = Math.min(Math.abs(d0), Math.abs(d1), Math.abs(d2));
    return !(sameSign && minAbsDistance > planeTolerance);
  };

  const collectSegmentsForTriangle = (triangle: Triangle): void => {
    const p0 = triangle[0];
    const p1 = triangle[1];
    const p2 = triangle[2];
    const points = [p0, p1, p2];
    const intersectionSegment: Point2[] = [];

    // Check each edge of the triangle for intersection
    for (let i = 0; i < 3; i++) {
      const triangleEdge = [points[i], points[(i + 1) % 3]] as LineSegment;

      // Check if the edge is nearly parallel and close to the plane
      const isNearlyParallelAndClose = isNearlyParallelAndCloseToPlane(
        triangleEdge,
        plane
      );
      if (isNearlyParallelAndClose) {
        const projectedSegment = toPlanarPoints(
          triangleEdge,
          plane
        ) as LineSegment2;
        intersectionLines.push(projectedSegment);

        continue;
      }

      const intersection = getSegmentPlaneIntersection(triangleEdge, plane);
      if (intersection) {
        intersectionSegment.push(intersection);
      }
    }

    // If there are two intersection points, add the line segment to the list
    if (intersectionSegment.length === 2) {
      intersectionLines.push(intersectionSegment as LineSegment2);
    }
  };

  for (const triangle of elementTriangles) {
    if (!triangleIntersectsPlane(triangle)) {
      continue;
    }
    collectSegmentsForTriangle(triangle);
  }

  return intersectionLines;
}

function findPolygonContainingPoint(
  magicShapeResult: NonNullable<MagicShapeResult>,
  polygons: Polygon2[]
): Polygon2 | undefined {
  return polygons.find((polygon) => {
    const planarHitPoint = toPlanarPoint(
      [
        magicShapeResult.hitTest.point.x,
        magicShapeResult.hitTest.point.y,
        magicShapeResult.hitTest.point.z,
      ],
      magicShapeResult.plane
    );

    return isPointWithinRing(polygon.exterior, planarHitPoint);
  });
}
