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

package org.apache.spark.mllib.feature;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import mtree.ComposedSplitFunction;
import mtree.DistanceFunction;
import mtree.DistanceFunctions;
import mtree.MTree;
import mtree.MTree.Query;
import mtree.MTree.ResultItem;
import mtree.PartitionFunctions;
import mtree.PromotionFunction;
import mtree.DistanceFunctions.EuclideanCoordinate;
import mtree.DataLP;
import mtree.utils.Pair;
import mtree.utils.Utils;


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
public class MTreeWrapper extends MTree<DataLP> implements Serializable {

	private static final long serialVersionUID = -4784817130749649598L;
	private int leafCount;
	private HashSet<DataLP> elements;
	
	private static final PromotionFunction<DataLP> nonRandomPromotion = new PromotionFunction<DataLP>() {
		@Override
		public Pair<DataLP> process(Set<DataLP> dataSet, DistanceFunction<? super DataLP> distanceFunction) {
			return Utils.minMax(dataSet);
		}
	};
	
	public MTreeWrapper () {
		super(2, DistanceFunctions.EUCLIDEAN, 
				new ComposedSplitFunction<DataLP>(
					nonRandomPromotion,
					new PartitionFunctions.BalancedPartition<DataLP>()
				)
			);		
		elements = new HashSet<DataLP>();
		leafCount = 0;
	}
	
	public MTreeWrapper (DataLP[] bulk) {
		this();
		/** Add elements to the iterator **/
		this.bulkInsert(bulk);
	}
	
	public boolean insert(DataLP pa) {		
		if(!elements.contains(pa)) {			
			this.add(pa);
			elements.add(pa);
			leafCount++;
			return true;
		}
		return false;		
	}
	
	public void insert(float[] point, float label) {
		DataLP pa = new DataLP(point, label);
		this.insert(pa);
	}
	
	/* The array must be sorted */
	public void bulkInsert(DataLP[] bulk){					
		for(int i = 0; i < bulk.length; i++) this.insert(bulk[i]);
	}
	
	public boolean remove(DataLP point) {
		if(elements.remove(point)) {
			super.remove(point);
		}		
		return false;
	}
	
	public int getSize() { return leafCount; }
	
	public Iterator<DataLP> getIterator() { return elements.iterator(); }
	
	
	public List<ResultItem> kNNQuery(int k, float[] point) {
		List<ResultItem> results = new ArrayList<ResultItem>();
		Query query = getNearestByLimit(new DataLP(point), k);
		for(ResultItem ri : query) {
			results.add(ri);
		}
		return results;
	}
	
	public List<ResultItem> overlapQuery(float[] point, double radius) {
		List<ResultItem> results = new ArrayList<ResultItem>();
		Query query = getNearestByRange(new DataLP(point), radius);
		for(ResultItem ri : query) {
			results.add(ri);
		}
		return results;
	}
	
	public List<ResultItem> overlapQuery(float[] point) {
		return this.overlapQuery(point, 0.0);
	}
	
	public List<ResultItem> searchIndices(float[] point, double tau) {
		return this.getIndices(new DataLP(point), tau);
	}
}
