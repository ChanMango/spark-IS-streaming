package mtree;

import java.io.Serializable;
import java.util.Arrays;

import mtree.DistanceFunctions.EuclideanCoordinate;

/**
 * @author sramirez
 */

public class DataLP implements EuclideanCoordinate, Comparable<DataLP>, Serializable {

	private static final long serialVersionUID = 1L;
	private final float[] values;
	private final float label;
  
  public DataLP(float[] values, float label) {
    this.values = values;
    this.label = label;
  }
  
  public DataLP(float[] values) {
    this.values = values;
    this.label = -1;
  }
  
  public float getLabel(){ return label;}
  public float[] getFeatures(){ return values;}
  
  
  @Override
  public int hashCode() {
	  return Arrays.hashCode(values);
  }
  
  @Override
  public int dimensions() {
    return values.length;
  }

  @Override
  public double get(int index) {
    return values[index];
  }
  
  @Override
  public boolean equals(Object obj) {
    if(obj instanceof DataLP) {
    	DataLP that = (DataLP) obj;
      if(this.dimensions() != that.dimensions()) {
        return false;
      }
      for(int i = 0; i < this.dimensions(); i++) {
        if(this.values[i] != that.values[i]) {
          return false;
        }
      }
      return true;
    } else {
      return false;
    }
  }
  
  @Override
  public int compareTo(DataLP that) {
    int dimensions = Math.min(this.dimensions(), that.dimensions());
    for(int i = 0; i < dimensions; i++) {
      float v1 = this.values[i];
      float v2 = that.values[i];
      if(v1 > v2) {
        return +1;
      }
      if(v1 < v2) {
        return -1;
      }
    }
    
    if(this.dimensions() > dimensions) {
      return +1;
    }
    
    if(that.dimensions() > dimensions) {
      return -1;
    }
    
    return 0;
  }
  
}