package mtree;

import java.io.Serializable;

/**
 * @author sramirez
 */

public class IndexedLP extends DataLP implements Serializable {

	private static final long serialVersionUID = 1L;
	private final int index;
  
  public IndexedLP(float[] values, float label, int index) {
	super(values, label);
    this.index = index;
  }
  
  public IndexedLP(float[] values, int index) {
  	super(values);
    this.index = index;
  }
  
  public int getIndex(){ return index; }
  
}