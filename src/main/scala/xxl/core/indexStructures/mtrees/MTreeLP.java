/* XXL: The eXtensible and fleXible Library for data processing

Copyright (C) 2000-2011 Prof. Dr. Bernhard Seeger
                        Head of the Database Research Group
                        Department of Mathematics and Computer Science
                        University of Marburg
                        Germany

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library;  If not, see <http://www.gnu.org/licenses/>. 

    http://code.google.com/p/xxl/

*/

package xxl.core.indexStructures.mtrees;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.TreeMap;

import xxl.core.collections.containers.Container;
import xxl.core.collections.containers.MapContainer;
import xxl.core.collections.queues.DynamicHeap;
import xxl.core.cursors.Cursor;
import xxl.core.cursors.filters.Taker;
import xxl.core.cursors.sorters.MergeSorter;
import xxl.core.functions.AbstractFunction;
import xxl.core.functions.Constant;
import xxl.core.functions.Function;
import xxl.core.indexStructures.MTree;
import xxl.core.indexStructures.ORTree;
import xxl.core.indexStructures.SortBasedBulkLoading;
import xxl.core.indexStructures.Sphere;
import xxl.core.indexStructures.Tree.Query.Candidate;
import xxl.core.io.converters.ConvertableConverter;
import xxl.core.io.converters.Converter;
import xxl.core.spatial.points.LabeledPoint;


/**
 * Creates a flexible, disk resident <b>M-Tree</b> or <b>Slim-Tree</b> and performs exact match, range
 * and nearest neighbor queries on it. Furthermore a detailed performance
 * evaluation is implemented, so the time for building and filling
 * the complete M-Tree, as well I/O-operations to external memory and buffers,
 * as the time for the different query evaluations is determined. <br>
 * <p>
 * <b>Data:</b>
 * The M-Tree indexes 10000 entries of type <tt>DoublePoint</tt>, a high dimensional point, which
 * are extracted from the dataset rr_small.bin lying in the directory
 * xxl\data. rr_small.bin contains a sample of minimal bounding rectangles (mbr) from
 * railroads in Los Angeles, i.e., the entries are of type KPE. So all
 * entries have to be converted to DoublePoints at first, using the
 * mapping functions and factory methods provided by this class.
 * Another dataset, called st_small.bin, located in the same directory
 * is used for query evaluation. It also contains KPE objects, but
 * they represent mbr's of streets.
 * <p>
 * <b>Insertion:</b>
 * Elements delivered by input cursor iterating over the external dataset
 * rr_small.bin can be inserted into the M-Tree using two different
 * strategies: <br>
 * <ul>
 * <li> tuple insertion: each element of the input cursor is inserted separately </li>
 * <li> bulk insertion: quantities of elements are inserted at once </li>
 * </ul>
 * When using sort-based bulk insertion the user is able to choose different
 * compare modes: given order, by x-axis, by peano-value or by hilbert-value.
 * <p>
 * <b>Queries:</b>
 * <ul>
 * <li> exact match queries: <br>
 *		1000 exact match queries are performed, taking 1000 KPEs from
 *		st_small.bin, converting them to DoublePoints and querying
 *		target level 0 of the M-Tree, i.e., only leaf nodes will be
 *		returned.
 * </li>
 * <li> range queries: <br>
 * 		1000 range queries are performed, taking 1000 KPEs from
 *		st_small.bin and converting them to Spheres, such that
 *		a sphere covers the rectangle belonging to a KPE, i.e.,
 *		the center of the sphere is set to the center of the rectangle.
 *		These spheres represent descriptors. The descriptors
 *		are used for the query at any specified target level in the M-Tree.
 * </li>
 * <li> nearest neighbor queries: <br>
 * 		A nearest neighbor query is also executed. Therefore a
 * 		sphere resulting by applying a mapping function to
 * 		an input element delivered from the input cursor iterating
 *		over st_small.bin is used as a query object, to which
 * 		the fifty nearest neighbors will be determined. For this
 *		realization a priority queue (dynamic heap) based on a special comparator
 *		defining an order on the distances of arbitrary points to
 * 		the query object is made use of.
 * </li>
 * </ul>
 * <p>
 * <b>Parameters:</b>
 * <ul>
 * <li> 1.) minimum capacity of nodes </li>
 * <li> 2.) maximum capacity of nodes </li>
 * <li> 3.) insertion type: tuple or bulk (different compare modes) </li>
 * <li> 4.) buffersize (number of node-objects)</li>
 * <li> 5.) target level for queries. Default: 0 </li>
 * <li> 6.) split strategy
 * <li> 7.) type of tree: MTree or SlimTree
 * </ul>
 * <p>
 * <b>Example usage:</b>
 * <pre>
 * 	java -Dxxlrootpath=W:\\dev\\xxl.cvs\\xxl -Dxxloutpath=W:\\dev\\xxl.out\\ xxl.applications.indexStructures.MTreeTest minCap=10 maxCap=25 insertType=bulk_hilbert bufSize=256 level=0 slimTree
 *
 * 	or:
 *
 * 	xxl xxl.applications.indexStructures.MTreeTest minCap=10 maxCap=25 insertType=bulk_hilbert bufSize=256 level=0 slimTree
 *
 * </pre>
 * For further parameters and settings of default values type:
 * <pre>
 *	java xxl.applications.indexStructures.MTreeTest /?
 * </pre>
 *
 *
 * @see java.util.Iterator
 * @see java.util.Comparator
 * @see xxl.core.collections.containers.Container
 * @see xxl.core.functions.Function
 * @see xxl.core.indexStructures.MTree
 * @see xxl.core.indexStructures.ORTree
 * @see xxl.core.indexStructures.SlimTree
 * @see xxl.core.indexStructures.Sphere
 * @see xxl.core.indexStructures.Tree
 * @see xxl.core.io.converters.Converter
 * @see xxl.core.io.LRUBuffer
 * @see xxl.core.spatial.points.DoublePoint
 * @see xxl.core.spatial.KPE
 * @see xxl.core.spatial.points.Point
 * @see xxl.core.spatial.rectangles.Rectangle
 */
public class MTreeLP implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4784817130749649598L;
	private MTree tree;
	private int leafCount = 0;
	private ArrayList<LabeledPoint> elements;
	

	/**
	 * A factory method to be invoked with one parameter of type KPE
	 * and returning an instance of the class Sphere covering the
	 * whole rectangle.
	 */
	/*public static Function SPHERE_COVERING_RECTANGLE_FACTORY = new AbstractFunction() {
		public Object invoke (Object kpe) {
			FloatPoint center = getMidPoint((FloatPoint)kpe);
			return new Sphere (center, center.distanceTo(((Rectangle)((KPE)kpe).getData()).getCorner(true)),
						   centerConverter(center.dimensions()));
		}
	};*/

	/**
	 * An unary factory method that returns a descriptor for
	 * a given point. That means a new Sphere is generated
	 * containing the given point as its center.
	 */
	public static Function getDescriptor = new AbstractFunction() {
		public Object invoke (Object o) {
			LabeledPoint point = (LabeledPoint)o;
			return new Sphere(point, 0.0, centerConverter(point.dimensions()));
		}
	};

	/**
	 * Returns a converter that serializes the center
	 * of sphere objects. In this case the center of a sphere
	 * is an high-dimensional DoublePoint. <br>
	 * This converter is used by the M-Tree to read/write leaf
	 * nodes to external memory.
	 *
	 * @param dimension the dimension of the DoublePoint representing
	 * 		the center of the sphere
	 * @return a converter serializing DoublePoints
	 */
	public static Converter centerConverter (final int dimension) {
		return new ConvertableConverter(
			new AbstractFunction() {
				public Object invoke () {
					return new LabeledPoint(dimension);
				}
			}
		);
	}
	

	/**
	 * A factory method returning a Comparator defining an order on points
	 * according to their first dimension. <br>
	 * This is only needed for sort-based bulk insertion.
	 */
	static Comparator COMPARE_X_AXIS = new Comparator() {
		public int compare (Object point1, Object point2) {
			return ((LabeledPoint)point1).getValue(0) < ((LabeledPoint)point2).getValue(0) ? -1 :
				   ((LabeledPoint)point1).getValue(0) == ((LabeledPoint)point2).getValue(0) ? 0 : 1;
		}
	};

	/**
	 * Returns a converter that serializes the descriptors of
	 * the M-Tree, i.e., spheres
	 * This converter is used by the M-Tree to read/write index
	 * nodes to external memory
	 *
	 * @param dimension the dimension of the DoublePoint representing
	 * 		the center of the sphere
	 * @return a converter serializing spheres
	 */
	public static Converter descriptorConverter (final int dimension) {		
		return new ConvertableConverter(
			new AbstractFunction() {
				public Object invoke () {
					return new Sphere(new LabeledPoint(dimension), 0.0, centerConverter(dimension));
				}
			}
		);
	}

	/**
	 * Returns a comparator which evaluates the distance of two candidate objects
	 * to the specified <tt>queryObject</tt>. This comparator
	 * is used for nearest neighbor queries and defines an order on candidate-
	 * descriptors. With the help of a priority queue (Min-heap) and this
	 * comparator the nearest neighbor query can be performed.
	 *
	 * @param queryObject a sphere to which the nearest neighbors should be determined
	 * @return a comparator defining an order on candidate objects
	 */
	public static Comparator getDistanceBasedComparator (final Sphere queryObject) {
		return new Comparator () {
			public int compare (Object candidate1, Object candidate2) {
				double sphereDist1 = queryObject.sphereDistance((Sphere)((Candidate)candidate1).descriptor()),
					   sphereDist2 = queryObject.sphereDistance((Sphere)((Candidate)candidate2).descriptor());
				return sphereDist1 < sphereDist2 ? -1 : sphereDist1 == sphereDist2 ? 0 : 1;
			}
		};
	}

	
	public MTreeLP () {

		int minCapacity = 10;
		int maxCapacity = 25;
		// generating a new instance of the class M-Tree or Slim-Tree
		tree = new MTree(MTree.HYPERPLANE_SPLIT);

		// initialize the MTree with the descriptor-factory method, a
		// container for storing the nodes and their minimum and maximum
		// capacity
		tree.initialize(getDescriptor, new MapContainer(new TreeMap()), minCapacity, maxCapacity);
		elements = new ArrayList<LabeledPoint>();		
	}
	
	public MTreeLP (LabeledPoint[] bulk) {

		int minCapacity = 10;
		int maxCapacity = 25;
		// generating a new instance of the class M-Tree or Slim-Tree
		tree = new MTree(MTree.HYPERPLANE_SPLIT);

		// initialize the MTree with the descriptor-factory method, a
		// container for storing the nodes and their minimum and maximum
		// capacity
		Container container = new MapContainer(new TreeMap());
		tree.initialize(getDescriptor, container, minCapacity, maxCapacity);
		
		/** Bulk-load all initial elements */
		Iterator<LabeledPoint> cursor = Arrays.asList(bulk).iterator();
		new SortBasedBulkLoading(tree, cursor, new Constant(container));

		/** Add elements to the iterator **/
		elements = new ArrayList<LabeledPoint>();
		for(int i = 0; i < bulk.length; i++) elements.add(bulk[i]);
		leafCount += bulk.length;
		
	}
	
	public void insert(float[] point, float label) {
		LabeledPoint pa = new LabeledPoint(point, label);
		tree.insert(pa);
		elements.add(pa);
		leafCount++;
	}
	
	public void insert(LabeledPoint lp) {
		tree.insert(lp);
		elements.add(lp);
		leafCount++;
	}
	
	/* The array must be sorted */
	public void bulkInsert(LabeledPoint[] bulk){		
		Iterator<LabeledPoint> cursor = Arrays.asList(bulk).iterator();
		new SortBasedBulkLoading(tree, cursor, tree.determineContainer);		
		for(int i = 0; i < bulk.length; i++) elements.add(bulk[i]);
		leafCount += bulk.length;
	}
	
	public boolean remove(float[] point) {
		LabeledPoint fpoint = new LabeledPoint(point);
		Object obj = tree.remove(fpoint);
		return obj != null;
	}
	
	public int getSize() { return leafCount; }
	
	public int getHeight() { return tree.height();}
	
	public Iterator<LabeledPoint> getIterator() { return tree.query(tree.rootDescriptor(), 0); }
	
	
	public List<Pair<Double, LabeledPoint>> kNNQuery(int k, float[] point) {		
		// consuming one further input element applying the mapping function
		// (KPE ==> Sphere) to it
		//Sphere queryObject = (Sphere)SPHERE_COVERING_RECTANGLE_FACTORY.invoke(new FloatPoint(point));
		LabeledPoint fpoint = new LabeledPoint(point);
		final Sphere queryObject = new Sphere(fpoint, 0.0, centerConverter(fpoint.dimensions()));
		
		// consuming the k-nearest elements concerning the query object at the
		// the target level;
		// the order is determined by the comparator given to the dynamic heap
		// structure realizing a priority queue
		
		Cursor cursor = new Taker(
			tree.query(new DynamicHeap(getDistanceBasedComparator(queryObject))),
			k
		);
		
		List<Pair<Double, LabeledPoint>> output = new ArrayList<Pair<Double, LabeledPoint>>(); 
		while (cursor.hasNext()) {
			Sphere next = (Sphere)((Candidate)cursor.next()).descriptor();
			double distance = queryObject.centerDistance(next);
			LabeledPoint center = (LabeledPoint) next.center();
			output.add(new Pair<Double, LabeledPoint>(distance, center));
		}
		
		cursor.close();
		return output;
	}
	
	public List<LabeledPoint> overlapQuery(float[] point) {		
		LabeledPoint fpoint = new LabeledPoint(point);
		final Sphere queryObject = new Sphere(fpoint, 0.0, centerConverter(fpoint.dimensions()));
		/*The result is a lazy cursor pointing to all leaf entries whose descriptors 
		 * overlap with the given <tt>queryDescriptor</tt>. */
		Cursor cursor = tree.query(queryObject);
		
		List<LabeledPoint> output = new ArrayList<LabeledPoint>(); 
		while (cursor.hasNext()) {			
			output.add((LabeledPoint) cursor.next());
		}		
		cursor.close();
		return output;
	}
	
	public List<LabeledPoint> overlapQuery(float[] point, double tau) {		
		LabeledPoint fpoint = new LabeledPoint(point);
		final Sphere queryObject = new Sphere(fpoint, tau, centerConverter(fpoint.dimensions()));
		/*The result is a lazy cursor pointing to all leaf entries whose descriptors 
		 * overlap with the given <tt>queryDescriptor</tt>. */
		Cursor cursor = tree.query(queryObject);
		
		List<LabeledPoint> output = new ArrayList<LabeledPoint>(); 
		while (cursor.hasNext()) {			
			output.add((LabeledPoint) cursor.next());
		}		
		cursor.close();
		return output;
	}
	
	public class Pair<Double, LabeledPoint> {

		  private final Double distance;
		  private final LabeledPoint point;

		  public Pair(Double distance, LabeledPoint point) {
		    this.distance = distance;
		    this.point = point;
		  }

		  public Double getDistance() { return distance; }
		  public LabeledPoint getPoint() { return point; }

		  @Override
		  public int hashCode() { return distance.hashCode() ^ point.hashCode(); }

		  @Override
		  public boolean equals(Object o) {
		    if (!(o instanceof Pair)) return false;
		    Pair pairo = (Pair) o;
		    return this.distance.equals(pairo.getDistance()) &&
		           this.point.equals(pairo.getPoint());
		  }

	}
	
	
    private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException{
        //ois.defaultReadObject();

        //notice the order of read and write should be same
    	int height = ois.readInt();
        leafCount = ois.readInt();
        elements = (ArrayList) ois.readObject();
        Sphere rd = (Sphere) ois.readObject();
        Object rpi = ois.readObject();
        //MapContainer container = (MapContainer) ois.readObject();
        
		// generating a new instance of the class M-Tree or Slim-Tree
		final MTree mTree = new MTree(MTree.HYPERPLANE_SPLIT);
		if(rpi != null && rd != null) {
			ORTree.IndexEntry rootEntry = (ORTree.IndexEntry) ((ORTree.IndexEntry)mTree.createIndexEntry(height)).initialize(rd).initialize(rpi);
			mTree.initialize(rootEntry, MTreeLP.getDescriptor, new MapContainer(new TreeMap()), 10, 25);
		} else {
			mTree.initialize(MTreeLP.getDescriptor, new MapContainer(new TreeMap()), 10, 25);
		}
		tree = mTree;         
    }
     
    private void writeObject(ObjectOutputStream oos) throws IOException {		
        //oos.defaultWriteObject();
        
		Object rootPageId = null;
		Sphere rootDescriptor = null;
        if(leafCount > 0) {
        	rootPageId = tree.rootEntry().id();   
        	rootDescriptor = (Sphere) tree.rootDescriptor();
        }
        
        oos.writeInt(tree.height());
        oos.writeInt(leafCount);
        oos.writeObject(elements);
        oos.writeObject(rootDescriptor);
        oos.writeObject(rootPageId);
        //oos.writeObject(container);
    }
}
